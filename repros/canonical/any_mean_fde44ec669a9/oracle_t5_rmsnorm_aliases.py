"""Gap diagnosis: this oracle computes the full captured T5 inference RMSNorm path, including fp16 residual add, global any(isinf) clamp-bound selection, clamp, fp16 cast, mean(square), rsqrt eps=1e-6, fp16 weight multiply, and three aliasing [2048,512] view outputs over one final buffer, whereas Inductor still schedules the scalar isinf predicate and row RMSNorm epilogue as generic reductions and pointwise work with extra launch/scalar-dependency overhead; Inductor cannot do this today because its normalization templates do not recognize the T5 clamp-bound RMSNorm pattern with metadata-only multi-output alias returns; the fix is NEW_PATTERN: add a T5 RMSNorm lowering that preserves the global finite-bound clamp choice, row RMSNorm, fp16 weight epilogue, and alias-view output contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 2048
HIDDEN = 512
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (1, ROWS, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
VIEW_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-6

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _row_inf_flags_kernel(
        mm_ptr,
        residual_ptr,
        row_flags_ptr,
        hidden: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        added_f16 = (mm + residual).to(tl.float16)
        has_inf = tl.abs(added_f16) == float("inf")
        row_has_inf = tl.max(tl.where(mask & has_inf, 1, 0), axis=0)
        tl.store(row_flags_ptr + row, row_has_inf.to(tl.int8))

    @triton.jit
    def _reduce_inf_flags_kernel(
        row_flags_ptr,
        any_inf_ptr,
        rows: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_M)
        mask = offsets < rows
        flags = tl.load(row_flags_ptr + offsets, mask=mask, other=0).to(tl.int32)
        any_inf = tl.max(flags, axis=0)
        tl.store(any_inf_ptr, any_inf.to(tl.int8))

    @triton.jit
    def _t5_rmsnorm_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        any_inf_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        added_f16 = (mm + residual).to(tl.float16)
        added_f32 = added_f16.to(tl.float32)

        any_inf = tl.load(any_inf_ptr).to(tl.int32) != 0
        bound = tl.where(any_inf, 64504.0, 65504.0)
        clamped_f32 = tl.where(
            added_f32 != added_f32,
            added_f32,
            tl.minimum(tl.maximum(added_f32, -bound), bound),
        )
        clamped_f16 = clamped_f32.to(tl.float16)
        x = clamped_f16.to(tl.float32)

        square = tl.where(mask, x * x, 0.0)
        mean_square = tl.sum(square, axis=0) / hidden
        inv_rms = tl.rsqrt(mean_square + eps)

        norm_f16 = (x * inv_rms).to(tl.float16)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        out = (norm_f16 * weight).to(tl.float16)
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm_85, residual, weight, shape0, shape1, shape2, shape3 = inputs
    mm = _require_tensor("mm_85", mm_85, INPUT_SHAPE, torch.float16)
    residual_t = _require_tensor("convert_element_type_142", residual, RESIDUAL_SHAPE, torch.float16)
    weight_t = _require_tensor("arg118_1", weight, WEIGHT_SHAPE, torch.float16)

    if residual_t.device != mm.device or weight_t.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    _require_shape("_shape_param_0", shape0, RESIDUAL_SHAPE)
    out_shape1 = _require_shape("_shape_param_1", shape1, VIEW_SHAPE)
    out_shape2 = _require_shape("_shape_param_2", shape2, VIEW_SHAPE)
    out_shape3 = _require_shape("_shape_param_3", shape3, VIEW_SHAPE)

    return mm, residual_t, weight_t, out_shape1, out_shape2, out_shape3


@oracle_impl(hardware="H100", shapes="(T([2048, 512], f16), T([1, 2048, 512], f16), T([512], f16), S([1, 2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the full Repro.forward T5 RMSNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same three fp16 [2048,512] view outputs. The returned views alias the
    same base storage, matching the eager `view(mul_tensor_1, shape)` contract.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_t5_rmsnorm_aliases.py")

    mm, residual, weight, out_shape1, out_shape2, out_shape3 = _validate_inputs(inputs)
    row_flags = torch.empty((ROWS,), device=mm.device, dtype=torch.int8)
    any_inf = torch.empty((1,), device=mm.device, dtype=torch.int8)
    base = torch.empty_strided(
        RESIDUAL_SHAPE,
        (ROWS * HIDDEN, HIDDEN, 1),
        device=mm.device,
        dtype=torch.float16,
    )

    block_n = triton.next_power_of_2(HIDDEN)
    _row_inf_flags_kernel[(ROWS,)](
        mm,
        residual,
        row_flags,
        hidden=HIDDEN,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )
    _reduce_inf_flags_kernel[(1,)](
        row_flags,
        any_inf,
        rows=ROWS,
        BLOCK_M=triton.next_power_of_2(ROWS),
        num_warps=8,
        num_stages=3,
    )
    _t5_rmsnorm_kernel[(ROWS,)](
        mm,
        residual,
        weight,
        any_inf,
        base,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )

    return (base.view(out_shape1), base.view(out_shape2), base.view(out_shape3))


# --- CLI entry point ---
def main():
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
