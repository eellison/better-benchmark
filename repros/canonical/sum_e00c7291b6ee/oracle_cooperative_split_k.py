"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete wide-row bf16 softmax-backward output by splitting each row across column tiles, cooperatively reducing bf16-rounded probability partials, and folding row-summary finalization into the tiled output epilogue that recomputes the precision-rounded producer, whereas Inductor currently schedules the exp/div/bf16 round-trip producer, the row sum, and the post-reduction fma/cast as one generic wide-row reduction that serializes too much work inside each row program; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K template for extremely wide row reductions with a dependent full-tensor epilogue and precision-preserving producer recompute; the fix is COOPERATIVE_SPLIT_K: add a row-split softmax-backward lowering that emits tiled row-sum partials and fuses row-summary finalization plus the bf16 output epilogue without full-size f32 intermediates."""
from __future__ import annotations

import argparse
import importlib.util
import sys
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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_BLOCK_N = 32768
DEFAULT_OUTPUT_BLOCK_N = 1024


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

    @triton.jit
    def _softmax_backward_epilogue_from_partials_kernel(
        logits_ptr,
        row_max_ptr,
        row_denom_ptr,
        grad_scalar_ptr,
        partial_ptr,
        out_ptr,
        N_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        PARTIAL_BLOCK_: tl.constexpr,
        OUTPUT_BLOCK_N_: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)

        partial_tiles = tl.arange(0, PARTIAL_BLOCK_)
        partial_mask = partial_tiles < NUM_TILES_
        partials = tl.load(
            partial_ptr + row * NUM_TILES_ + partial_tiles,
            mask=partial_mask,
            other=0.0,
        ).to(tl.float32)
        prob_sum = tl.sum(partials, axis=0)
        grad = tl.load(grad_scalar_ptr).to(tl.float32)
        row_scale = grad * (1.0 - prob_sum)

        cols = tile * OUTPUT_BLOCK_N_ + tl.arange(0, OUTPUT_BLOCK_N_)
        mask = cols < N_
        offsets = row * N_ + cols
        row_max = tl.load(row_max_ptr + row).to(tl.float32)
        row_denom = tl.load(row_denom_ptr + row).to(tl.float32)
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
    _softmax_backward_epilogue_from_partials_kernel[(m, output_num_tiles)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        partials,
        out,
        N_=n,
        NUM_TILES_=num_tiles,
        PARTIAL_BLOCK_=partial_block,
        OUTPUT_BLOCK_N_=output_block_n,
        num_warps=4,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([], bf16), T([8192, 262144], bf16), T([8192, 1], f32), T([8192, 1], f32), S([8192, 262144]))")
def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Standard harness entry point: compute the full Repro.forward output."""
    return oracle_cooperative_split_k(*inputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
