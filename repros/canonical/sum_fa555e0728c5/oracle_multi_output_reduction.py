"""
Full-scope Triton oracle for sum_fa555e0728c5 (MobileBERT masked transpose + sum).

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle consumes the same
`mm_708`, boolean mask, scalar fill value, and shape parameters as repro.py,
materializes the returned `[512, 32768]` transpose-view backing storage while
accumulating the sibling `[512]` column sum from the same streamed `where`
producer, and returns outputs with the same dtypes, shapes, and strides.
Inductor cannot expose a meaningful remaining optimization for this captured
shape today because the best compiled variants already lower the full scope to
two bandwidth-dominated kernels with effectively the same memory traffic: read
`mm`, read the bool mask, write the full returned tensor, and finalize small
column partials. The fix class is BANDWIDTH_BOUND; no scheduler change is
justified unless a future full-scope oracle beats both required local compile
configs and the historical queue best.
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


REPRO_ID = "sum_fa555e0728c5"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 32768
C = 512
HISTORICAL_BEST_COMPILE_US = 33.76000002026558

DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 64
DEFAULT_FINAL_BLOCK_C = 16

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
def _where_store_partial_sum_kernel(
    mm_ptr,
    mask_ptr,
    full_ptr,
    out_base_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    active = (rows[:, None] < ROWS_) & (cols[None, :] < C_)
    offsets = rows[:, None] * C_ + cols[None, :]

    mm_values = tl.load(mm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mask_values = tl.load(mask_ptr + offsets, mask=active, other=0)
    full_value = tl.load(full_ptr).to(tl.float32)
    values = tl.where(mask_values, full_value, mm_values)

    tl.store(out_base_ptr + offsets, values, mask=active)
    partial = tl.sum(tl.where(active, values, 0.0), axis=0)
    tl.store(partial_ptr + row_block * C_ + cols, partial, mask=cols < C_)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    active = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (cols[None, :] < C_)
    offsets = row_blocks[:, None] * C_ + cols[None, :]

    partials = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    tl.store(sum_ptr + cols, tl.sum(partials, axis=0), mask=cols < C_)


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


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    if len(inputs) != 6:
        raise ValueError(f"expected 6 repro inputs, got {len(inputs)}")

    mm_708, arg1313_1, full_1, shape0, shape1, shape2 = inputs
    if not isinstance(mm_708, torch.Tensor) or not isinstance(arg1313_1, torch.Tensor):
        raise TypeError("first two inputs must be tensors")
    if not isinstance(full_1, torch.Tensor):
        raise TypeError("third input must be the scalar fill tensor")
    if tuple(shape0) != (256, 128, 512):
        raise ValueError(f"unexpected first shape parameter: {shape0}")
    if tuple(shape1) != (ROWS, C):
        raise ValueError(f"unexpected second shape parameter: {shape1}")
    if tuple(shape2) != (C,):
        raise ValueError(f"unexpected third shape parameter: {shape2}")

    if mm_708.shape != (ROWS, C):
        raise ValueError(f"unexpected mm_708 shape: {tuple(mm_708.shape)}")
    if arg1313_1.shape != (256, 128, C):
        raise ValueError(f"unexpected mask shape: {tuple(arg1313_1.shape)}")
    if full_1.shape != ():
        raise ValueError(f"unexpected full_1 shape: {tuple(full_1.shape)}")
    if mm_708.stride() != (C, 1):
        raise ValueError(f"unexpected mm_708 stride: {mm_708.stride()}")
    if arg1313_1.stride() != (128 * C, C, 1):
        raise ValueError(f"unexpected mask stride: {arg1313_1.stride()}")
    if mm_708.dtype != torch.float32 or full_1.dtype != torch.float32:
        raise ValueError("expected float32 mm_708 and full_1")
    if arg1313_1.dtype != torch.bool:
        raise ValueError(f"expected bool mask, got {arg1313_1.dtype}")
    if any(t.device.type != "cuda" for t in (mm_708, arg1313_1, full_1)):
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")


def oracle_fused(
    mm_708: torch.Tensor,
    arg1313_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    final_block_c: int = DEFAULT_FINAL_BLOCK_C,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = (mm_708, arg1313_1, full_1, _shape_param_0, _shape_param_1, _shape_param_2)
    _validate_inputs(inputs)

    num_row_blocks = triton.cdiv(ROWS, block_m)
    out_base = torch.empty((ROWS, C), device=mm_708.device, dtype=torch.float32)
    partial = torch.empty((num_row_blocks, C), device=mm_708.device, dtype=torch.float32)
    out_sum = torch.empty((C,), device=mm_708.device, dtype=torch.float32)

    _where_store_partial_sum_kernel[(num_row_blocks, triton.cdiv(C, block_n))](
        mm_708,
        arg1313_1,
        full_1,
        out_base,
        partial,
        ROWS_=ROWS,
        C_=C,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _finalize_sum_kernel[(triton.cdiv(C, final_block_c),)](
        partial,
        out_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=final_block_c,
        num_warps=8,
    )

    return out_base.permute(1, 0), out_sum


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return float(diff.max().item()), float(rel.max().item())


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        shape_ok = got.shape == expected.shape
        dtype_ok = got.dtype == expected.dtype
        stride_ok = got.stride() == expected.stride()
        close_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        output_ok = shape_ok and dtype_ok and stride_ok and close_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={close_ok} shape_match={shape_ok} "
            f"stride_match={stride_ok} dtype_match={dtype_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    rep: int,
    warmup: int,
    no_compile: bool,
    block_m: int,
    block_n: int,
    final_block_c: int,
) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    timings: dict[str, float] = {}
    with torch.no_grad():
        oracle_fused(
            *inputs,
            block_m=block_m,
            block_n=block_n,
            final_block_c=final_block_c,
        )
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(
                *inputs,
                block_m=block_m,
                block_n=block_n,
                final_block_c=final_block_c,
            ),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(
        "oracle_fused full-scope where materialize + column sum: "
        f"{oracle_us:.3f} us "
        f"(BLOCK_M={block_m}, BLOCK_N={block_n}, FINAL_BLOCK_C={final_block_c})"
    )

    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config)
                compiled_us = triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                ) * 1000.0
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_required_compile = min(
        (timings[label] for label, _ in COMPILE_CONFIGS if label in timings),
        default=float("inf"),
    )
    best_reference = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < best_reference
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if valid_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": round(oracle_us, 3),
                "best_required_compile_us": (
                    round(best_required_compile, 3)
                    if best_required_compile != float("inf")
                    else None
                ),
                "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
                "valid_floor": valid_floor,
                "classification": "BANDWIDTH_BOUND",
            }
        )
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-3)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M, help=argparse.SUPPRESS)
    parser.add_argument("--block-n", type=int, default=DEFAULT_BLOCK_N, help=argparse.SUPPRESS)
    parser.add_argument(
        "--final-block-c",
        type=int,
        default=DEFAULT_FINAL_BLOCK_C,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            rep=args.rep,
            warmup=args.warmup,
            no_compile=args.no_compile,
            block_m=args.block_m,
            block_n=args.block_n,
            final_block_c=args.final_block_c,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
