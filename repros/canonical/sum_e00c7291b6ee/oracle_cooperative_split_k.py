"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete wide-row bf16 softmax-backward output by splitting each row across column tiles, cooperatively reducing per-row bf16-rounded probability partials, finalizing one row summary per output row, and recomputing the precision-rounded producer in a parallel tiled epilogue, whereas Inductor currently schedules the exp/div/bf16 round-trip producer, the row sum, and the post-reduction fma/cast as one generic wide-row reduction that serializes too much work inside each row program; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K template for extremely wide row reductions with a dependent full-tensor epilogue and precision-preserving producer recompute; the fix is COOPERATIVE_SPLIT_K: add a row-split softmax-backward lowering that emits tiled row-sum partials, finalizes row summaries, and fuses the bf16 output epilogue without full-size f32 intermediates."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_e00c7291b6ee"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_BLOCK_N = 32768
DEFAULT_OUTPUT_BLOCK_N = 1024

sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _probability_partial_kernel(
        logits_ptr,
        row_max_ptr,
        row_denom_ptr,
        partial_ptr,
        N_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        mask = cols < N_
        offsets = row * N_ + cols

        x = tl.load(logits_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.load(row_max_ptr + row).to(tl.float32)
        row_denom = tl.load(row_denom_ptr + row).to(tl.float32)
        prob = (tl.exp(x - row_max) / row_denom).to(tl.bfloat16)
        prob_f32 = prob.to(tl.float32)

        partial = tl.sum(tl.where(mask, prob_f32, 0.0), axis=0)
        tl.store(partial_ptr + row * NUM_TILES_ + tile, partial)

    @triton.jit
    def _finalize_row_sum_kernel(
        grad_scalar_ptr,
        partial_ptr,
        row_scale_ptr,
        NUM_TILES_: tl.constexpr,
        PARTIAL_BLOCK_: tl.constexpr,
    ):
        row = tl.program_id(0)
        tiles = tl.arange(0, PARTIAL_BLOCK_)
        mask = tiles < NUM_TILES_
        partials = tl.load(partial_ptr + row * NUM_TILES_ + tiles, mask=mask, other=0.0)
        prob_sum = tl.sum(partials, axis=0)
        grad = tl.load(grad_scalar_ptr).to(tl.float32)
        tl.store(row_scale_ptr + row, grad * (1.0 - prob_sum))

    @triton.jit
    def _softmax_backward_epilogue_kernel(
        logits_ptr,
        row_max_ptr,
        row_denom_ptr,
        row_scale_ptr,
        out_ptr,
        N_: tl.constexpr,
        OUTPUT_BLOCK_N_: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * OUTPUT_BLOCK_N_ + tl.arange(0, OUTPUT_BLOCK_N_)
        mask = cols < N_
        offsets = row * N_ + cols
        row_max = tl.load(row_max_ptr + row).to(tl.float32)
        row_denom = tl.load(row_denom_ptr + row).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row).to(tl.float32)
        x = tl.load(logits_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        prob = (tl.exp(x - row_max) / row_denom).to(tl.bfloat16).to(tl.float32)
        out = prob * row_scale
        tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
    inputs = _load_repro_module().make_inputs()
    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _validate_inputs(
    arg3_1: torch.Tensor,
    arg0_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg2_1: torch.Tensor,
    shape_param: object,
) -> tuple[int, int]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if arg0_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if arg3_1.shape != ():
        raise ValueError(f"unexpected arg3_1 shape: {tuple(arg3_1.shape)}")
    if arg0_1.dtype != torch.bfloat16 or arg3_1.dtype != torch.bfloat16:
        raise ValueError("expected bf16 scalar grad and bf16 logits")
    if arg1_1.dtype != torch.float32 or arg2_1.dtype != torch.float32:
        raise ValueError("expected f32 row max and denominator")

    m, n = int(shape_param[0]), int(shape_param[1])
    if tuple(arg0_1.shape) != (m, n):
        raise ValueError(f"unexpected arg0_1 shape: got={tuple(arg0_1.shape)} expected={(m, n)}")
    if tuple(arg1_1.shape) != (m, 1) or tuple(arg2_1.shape) != (m, 1):
        raise ValueError("unexpected row max/denominator shape")
    if not arg0_1.is_contiguous() or not arg1_1.is_contiguous() or not arg2_1.is_contiguous():
        raise ValueError("oracle expects contiguous captured inputs")
    return m, n


def oracle_cooperative_split_k(
    arg3_1: torch.Tensor,
    arg0_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg2_1: torch.Tensor,
    shape_param: object,
    *,
    block_n: int = DEFAULT_BLOCK_N,
    output_block_n: int = DEFAULT_OUTPUT_BLOCK_N,
) -> torch.Tensor:
    """Compute the complete Repro.forward result with a cooperative split-K Triton path."""
    m, n = _validate_inputs(arg3_1, arg0_1, arg1_1, arg2_1, shape_param)
    if block_n <= 0 or (block_n & (block_n - 1)) != 0:
        raise ValueError("block_n must be a positive power of two")
    if output_block_n <= 0 or (output_block_n & (output_block_n - 1)) != 0:
        raise ValueError("output_block_n must be a positive power of two")

    num_tiles = triton.cdiv(n, block_n)
    output_num_tiles = triton.cdiv(n, output_block_n)
    partial_block = triton.next_power_of_2(num_tiles)
    out = torch.empty_strided((m, n), (n, 1), device=arg0_1.device, dtype=torch.bfloat16)
    partials = torch.empty((m, num_tiles), device=arg0_1.device, dtype=torch.float32)
    row_scales = torch.empty((m,), device=arg0_1.device, dtype=torch.float32)

    grid = (m, num_tiles)
    _probability_partial_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        partials,
        N_=n,
        NUM_TILES_=num_tiles,
        BLOCK_N_=block_n,
        num_warps=4,
    )
    _finalize_row_sum_kernel[(m,)](
        arg3_1,
        partials,
        row_scales,
        NUM_TILES_=num_tiles,
        PARTIAL_BLOCK_=partial_block,
        num_warps=8,
    )
    _softmax_backward_epilogue_kernel[(m, output_num_tiles)](
        arg0_1,
        arg1_1,
        arg2_1,
        row_scales,
        out,
        N_=n,
        OUTPUT_BLOCK_N_=output_block_n,
        num_warps=4,
    )
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _chunked_reference_check(
    actual: torch.Tensor,
    inputs: tuple[object, ...],
    *,
    chunk_n: int,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float]:
    arg3_1, arg0_1, arg1_1, arg2_1, shape_param = inputs
    m, n = int(shape_param[0]), int(shape_param[1])
    grad = arg3_1.float()
    row_sum = torch.zeros((m, 1), device=arg0_1.device, dtype=torch.float32)

    for start in range(0, n, chunk_n):
        end = min(start + chunk_n, n)
        probs = torch.exp(arg0_1[:, start:end].float() - arg1_1) / arg2_1
        probs = probs.to(torch.bfloat16).float()
        row_sum += (grad * probs).sum(dim=-1, keepdim=True)

    max_abs = 0.0
    max_rel = 0.0
    ok = True
    for start in range(0, n, chunk_n):
        end = min(start + chunk_n, n)
        probs = torch.exp(arg0_1[:, start:end].float() - arg1_1) / arg2_1
        probs = probs.to(torch.bfloat16).float()
        expected = (grad * probs - probs * row_sum).to(torch.bfloat16)
        diff = (actual[:, start:end].float() - expected.float()).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        rel = diff / (expected.float().abs() + 1.0e-8)
        max_rel = max(max_rel, float(rel.max().item()))
        ok = ok and bool((diff <= (atol + rtol * expected.float().abs())).all().item())
        del probs, expected, diff, rel
    return ok, max_abs, max_rel


def run_check(
    device: torch.device,
    rtol: float,
    atol: float,
    block_n: int,
    output_block_n: int,
) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        actual = oracle_cooperative_split_k(
            *inputs,
            block_n=block_n,
            output_block_n=output_block_n,
        )
        synchronize(device)

    arg0_1 = inputs[1]
    shape_ok = actual.shape == arg0_1.shape
    dtype_ok = actual.dtype == torch.bfloat16
    stride_ok = actual.stride() == arg0_1.stride()
    value_ok, max_abs, max_rel = _chunked_reference_check(
        actual,
        inputs,
        chunk_n=block_n,
        rtol=rtol,
        atol=atol,
    )
    ok = shape_ok and dtype_ok and stride_ok and value_ok
    print(
        f"output[0]: shape={list(actual.shape)} dtype={actual.dtype} stride={actual.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} shape_match={shape_ok} "
        f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    if triton is not None and device.type == "cuda":
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)
    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def _compile_repro(inputs: tuple[object, ...]) -> torch.nn.Module:
    import torch._dynamo

    torch._dynamo.reset()
    model = _load_repro_module().Repro().eval()
    compiled = torch.compile(model)
    with torch.no_grad():
        compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    device: torch.device,
    warmup: int,
    rep: int,
    block_n: int,
    output_block_n: int,
    skip_compile: bool,
) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    _, arg0_1, _, _, shape_param = inputs
    m, n = int(shape_param[0]), int(shape_param[1])
    num_tiles = triton.cdiv(n, block_n) if triton is not None else math.ceil(n / block_n)
    output_num_tiles = triton.cdiv(n, output_block_n) if triton is not None else math.ceil(n / output_block_n)
    logical_bytes = (2 * m * n * 2) + (m * n * 2) + (m * num_tiles * 4) + (m * 4)
    print(
        f"oracle shape: bf16[{m}, {n}] block_n={block_n} "
        f"output_block_n={output_block_n} num_tiles={num_tiles} "
        f"output_num_tiles={output_num_tiles} device={device}"
    )
    print(f"approx oracle traffic: {logical_bytes / 1e9:.3f} GB")

    with torch.no_grad():
        oracle_cooperative_split_k(
            *inputs,
            block_n=block_n,
            output_block_n=output_block_n,
        )
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_cooperative_split_k(
                *inputs,
                block_n=block_n,
                output_block_n=output_block_n,
            ),
            device,
            warmup,
            rep,
        )
    print(f"oracle_cooperative_split_k: {oracle_us:.3f} us warmup={warmup} rep={rep}")

    if skip_compile:
        return

    compiled = _compile_repro(inputs)
    with torch.no_grad():
        compile_us = benchmark(lambda: compiled(*inputs), device, warmup, rep)
    print(f"torch.compile default: {compile_us:.3f} us")
    if oracle_us > 0:
        print(f"speedup_vs_compile: {compile_us / oracle_us:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle output with a chunked Repro-equivalent reference")
    parser.add_argument("--bench", action="store_true", help="benchmark the Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument("--output-block-n", type=int, default=DEFAULT_OUTPUT_BLOCK_N)
    parser.add_argument("--skip-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(
        device,
        args.rtol,
        args.atol,
        args.block_n,
        args.output_block_n,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            device,
            args.warmup,
            args.rep,
            args.block_n,
            args.output_block_n,
            args.skip_compile,
        )


if __name__ == "__main__":
    main()
