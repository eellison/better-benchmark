"""Full-scope Triton oracle for sum_80634958a126.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle consumes the same
four float32 inputs and five shape parameters as repro.py, computes the full
three-add producer, returns the same transposed f32[4096, 4096] view with
stride (1, 4096), and returns the same f32[4096] column sum. It differs from
Inductor by explicitly sharing the add producer between the materialized
transpose backing storage and the reduction: one Triton kernel writes the
contiguous add result and per-column partial sums, and a second Triton kernel
finalizes the vector. Inductor cannot express this exact schedule today because
its generic scheduler does not form a single producer-sharing template that
both materializes the backing storage for a returned non-contiguous view and
reduces that producer. The relevant Inductor change would be a scheduler/codegen
template for materialized-view plus reduction producer sharing, but this repro
is gated by the historical best: if the full-scope Triton path does not beat
both required local compile configs and historical best_compile_us, the result
is diagnosis-only/BANDWIDTH_BOUND rather than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_80634958a126"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 512
HIDDEN = 4096
ROWS = BATCH * SEQ
COLS = HIDDEN
THREE_D_SHAPE = (BATCH, SEQ, HIDDEN)
TWO_D_SHAPE = (ROWS, COLS)
CONTIG_2D_STRIDE = (COLS, 1)
TRANSPOSE_STRIDE = (1, COLS)

HISTORICAL_BEST_COMPILE_US = 69.2799985408783
CLASSIFICATION = "BANDWIDTH_BOUND"
DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 32
DEFAULT_CANDIDATES = "32x32,32x64,64x16,64x32,128x16"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _add_and_partial_sum_kernel(
    mm_142_ptr,
    mul_341_ptr,
    mm_144_ptr,
    mm_146_ptr,
    out_base_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * COLS_ + cols[None, :]
    mask = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)

    mul_341 = tl.load(mul_341_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    value = mul_341 + tl.load(
        mm_142_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    value += tl.load(
        mm_144_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    value += tl.load(
        mm_146_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    tl.store(out_base_ptr + offsets, value, mask=mask)

    row_mask = rows < ROWS_
    col_mask = cols < COLS_
    partial = tl.sum(tl.where(row_mask[:, None], value, 0.0), axis=0)
    partial_offsets = row_block * COLS_ + cols
    tl.store(partials_ptr + partial_offsets, partial, mask=col_mask)


@triton.jit
def _finalize_sum_kernel(
    partials_ptr,
    sum_out_ptr,
    COLS_: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = row_blocks[:, None] * COLS_ + cols[None, :]
    mask = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (cols[None, :] < COLS_)
    values = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(sum_out_ptr + cols, sums, mask=cols < COLS_)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x
        for x in module.make_inputs()
    )


def _validate_power_of_two(name: str, value: int) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")


def _validate_inputs(
    mm_142: torch.Tensor,
    mul_341: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
    shape_param_2: object,
    shape_param_3: object,
    shape_param_4: object,
) -> None:
    assert tuple(mm_142.shape) == TWO_D_SHAPE
    assert tuple(mul_341.shape) == THREE_D_SHAPE
    assert tuple(mm_144.shape) == TWO_D_SHAPE
    assert tuple(mm_146.shape) == TWO_D_SHAPE
    assert tuple(shape_param_0) == THREE_D_SHAPE
    assert tuple(shape_param_1) == THREE_D_SHAPE
    assert tuple(shape_param_2) == THREE_D_SHAPE
    assert tuple(shape_param_3) == TWO_D_SHAPE
    assert tuple(shape_param_4) == (COLS,)
    assert mm_142.dtype == torch.float32
    assert mul_341.dtype == torch.float32
    assert mm_144.dtype == torch.float32
    assert mm_146.dtype == torch.float32
    assert mm_142.is_contiguous()
    assert mul_341.is_contiguous()
    assert mm_144.is_contiguous()
    assert mm_146.is_contiguous()


def _num_warps_for_tile(block_m: int, block_n: int) -> int:
    elems = block_m * block_n
    if elems <= 1024:
        return 4
    return 8


def _num_warps_for_finalize(block_row_blocks: int, block_n: int) -> int:
    elems = block_row_blocks * block_n
    if elems <= 1024:
        return 4
    return 8


def _allocate_workspace(
    device: torch.device,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_row_blocks = triton.cdiv(ROWS, block_m)
    out_base = torch.empty(TWO_D_SHAPE, device=device, dtype=torch.float32)
    sum_out = torch.empty((COLS,), device=device, dtype=torch.float32)
    partials = torch.empty((num_row_blocks, COLS), device=device, dtype=torch.float32)
    return out_base, sum_out, partials


def _oracle_into(
    mm_142: torch.Tensor,
    mul_341: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    out_base: torch.Tensor,
    sum_out: torch.Tensor,
    partials: torch.Tensor,
    block_m: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert out_base.shape == TWO_D_SHAPE
    assert out_base.stride() == CONTIG_2D_STRIDE
    assert sum_out.shape == (COLS,)
    assert partials.shape == (triton.cdiv(ROWS, block_m), COLS)

    num_row_blocks = triton.cdiv(ROWS, block_m)
    grid = (num_row_blocks, triton.cdiv(COLS, block_n))
    _add_and_partial_sum_kernel[grid](
        mm_142,
        mul_341,
        mm_144,
        mm_146,
        out_base,
        partials,
        ROWS_=ROWS,
        COLS_=COLS,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=_num_warps_for_tile(block_m, block_n),
    )

    block_row_blocks = triton.next_power_of_2(num_row_blocks)
    _finalize_sum_kernel[(triton.cdiv(COLS, block_n),)](
        partials,
        sum_out,
        COLS_=COLS,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_ROW_BLOCKS=block_row_blocks,
        BLOCK_N=block_n,
        num_warps=_num_warps_for_finalize(block_row_blocks, block_n),
    )

    return out_base.t(), sum_out


def oracle_fused(
    mm_142: torch.Tensor,
    mul_341: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
    shape_param_2: object,
    shape_param_3: object,
    shape_param_4: object,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(
        mm_142,
        mul_341,
        mm_144,
        mm_146,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
        shape_param_4,
    )
    _validate_power_of_two("block_m", block_m)
    _validate_power_of_two("block_n", block_n)
    out_base, sum_out, partials = _allocate_workspace(mm_142.device, block_m)
    return _oracle_into(
        mm_142,
        mul_341,
        mm_144,
        mm_146,
        out_base,
        sum_out,
        partials,
        block_m,
        block_n,
    )


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Template-compatible full-scope oracle entry point."""
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float, block_m: int, block_n: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_fused(*inputs, block_m=block_m, block_n=block_n)
        torch.cuda.synchronize()

    if len(actual) != len(ref):
        print(f"output_count_match=False actual={len(actual)} expected={len(ref)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (actual_tensor, ref_tensor) in enumerate(zip(actual, ref, strict=True)):
        max_abs, max_rel = _max_diff(actual_tensor, ref_tensor)
        allclose = torch.allclose(actual_tensor.float(), ref_tensor.float(), rtol=rtol, atol=atol)
        shape_match = actual_tensor.shape == ref_tensor.shape
        stride_match = actual_tensor.stride() == ref_tensor.stride()
        dtype_match = actual_tensor.dtype == ref_tensor.dtype
        output_ok = allclose and shape_match and stride_match and dtype_match
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
            f"dtype={actual_tensor.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={allclose} shape_match={shape_match} stride_match={stride_match} "
            f"dtype_match={dtype_match}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _capture_graph_or_fallback(fn):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            return lambda: graph.replay(), "cudagraph"
        except Exception:
            return fn, "direct"


def _time_callable(fn, warmup: int, rep: int) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _parse_candidates(raw: str) -> tuple[tuple[int, int], ...]:
    candidates: list[tuple[int, int]] = []
    for item in raw.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if "x" not in text:
            raise ValueError("candidate entries must use BLOCK_MxBLOCK_N, for example 64x32")
        block_m_text, block_n_text = text.split("x", 1)
        block_m = int(block_m_text)
        block_n = int(block_n_text)
        _validate_power_of_two("block_m", block_m)
        _validate_power_of_two("block_n", block_n)
        candidates.append((block_m, block_n))
    if not candidates:
        raise ValueError("--candidates produced no tilings")
    return tuple(candidates)


def run_bench(
    rep: int,
    warmup: int,
    no_compile: bool,
    candidates: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    mm_142, mul_341, mm_144, mm_146, *shape_params = inputs
    _validate_inputs(mm_142, mul_341, mm_144, mm_146, *shape_params)

    oracle_times: list[tuple[int, int, float, str]] = []
    with torch.no_grad():
        for block_m, block_n in candidates:
            out_base, sum_out, partials = _allocate_workspace(mm_142.device, block_m)
            _oracle_into(
                mm_142,
                mul_341,
                mm_144,
                mm_146,
                out_base,
                sum_out,
                partials,
                block_m,
                block_n,
            )
            torch.cuda.synchronize()
            timed, timing_mode = _capture_graph_or_fallback(
                lambda block_m=block_m, block_n=block_n, out_base=out_base, sum_out=sum_out, partials=partials: _oracle_into(
                    mm_142,
                    mul_341,
                    mm_144,
                    mm_146,
                    out_base,
                    sum_out,
                    partials,
                    block_m,
                    block_n,
                )
            )
            oracle_us = _time_callable(timed, warmup=warmup, rep=rep)
            oracle_times.append((block_m, block_n, oracle_us, timing_mode))
            print(f"oracle_fused block_m={block_m} block_n={block_n}: {oracle_us:.3f} us ({timing_mode})")

    best_block_m, best_block_n, best_oracle_us, best_timing_mode = min(
        oracle_times,
        key=lambda item: item[2],
    )
    print(
        f"oracle_fused best full-scope add + transpose-view backing + column sum: "
        f"{best_oracle_us:.3f} us (block_m={best_block_m}, block_n={best_block_n}, {best_timing_mode})"
    )

    compile_times: list[tuple[str, float, str]] = []
    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            compiled = _compile_with_config(model, inputs, config)
            timed, timing_mode = _capture_graph_or_fallback(lambda compiled=compiled: compiled(*inputs))
            compiled_us = _time_callable(timed, warmup=warmup, rep=rep)
            compile_times.append((label, compiled_us, timing_mode))
            print(f"torch.compile {label}: {compiled_us:.3f} us ({timing_mode})")

    local_floor_ok = bool(compile_times) and all(best_oracle_us < compiled_us for _, compiled_us, _ in compile_times)
    historical_floor_ok = best_oracle_us < HISTORICAL_BEST_COMPILE_US
    true_floor = local_floor_ok and historical_floor_ok

    print(f"historical_best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"true_floor: {'yes' if true_floor else 'no'}")
    if not true_floor:
        print("diagnosis_only: oracle does not beat every required local compile config and historical best")

    result: dict[str, object] = {
        "repro_id": REPRO_ID,
        "classification": CLASSIFICATION,
        "oracle_us": best_oracle_us,
        "best_block_m": best_block_m,
        "best_block_n": best_block_n,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "true_floor": true_floor,
        "status": "TRUE_FLOOR" if true_floor else "DIAGNOSIS_ONLY",
    }
    for label, compiled_us, _ in compile_times:
        result[f"compile_{label}_us"] = compiled_us
    if compile_times:
        best_compile_us = min(compiled_us for _, compiled_us, _ in compile_times)
        result["best_required_compile_us"] = best_compile_us
        result["ratio_best_required_compile_to_oracle"] = best_compile_us / best_oracle_us
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N)
    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
        help="comma-separated BLOCK_MxBLOCK_N tilings for --bench",
    )
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    _validate_power_of_two("block_m", args.block_m)
    _validate_power_of_two("block_n", args.block_n)
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_m=args.block_m,
        block_n=args.block_n,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            rep=args.rep,
            warmup=args.warmup,
            no_compile=args.no_compile,
            candidates=_parse_candidates(args.candidates),
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
