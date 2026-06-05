"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 residual RMSNorm forward path in Triton, including the global any(isinf) clamp-bound choice, fp16-rounded clamp, fp32 mean(square), rsqrt eps=1e-6, fp16 weight multiply, final 0.04419417382415922 scale, and returned `[2048,512]` view, whereas tuned Inductor already lowers this fixed hidden-size norm region to the same mandatory memory-traffic envelope; Inductor cannot materially improve it through local fusion or algebraic rewrites because the remaining work is the required residual/input/weight reads, one row reduction, rsqrt, output store, and the scalar finite check rather than an avoidable intermediate materialization; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless a broader norm-template or launch-overhead optimization moves the baseline."""
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

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
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
ROWS = 2048
HIDDEN = 512
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (1, ROWS, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
TOTAL_ELEMENTS = ROWS * HIDDEN
BLOCK_ELEMS = 1024
BLOCK_H = 512
EPS = 1.0e-6
FINITE_BOUND = 65504.0
INF_BOUND = 64504.0
FINAL_SCALE = 0.04419417382415922

if triton is not None:

    @triton.jit
    def _clear_flag_kernel(flag_ptr):
        tl.store(flag_ptr, 0)

    @triton.jit
    def _any_inf_kernel(
        mm_ptr,
        residual_ptr,
        flag_ptr,
        n_elements: tl.constexpr,
        block_elems: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_elems + tl.arange(0, block_elems)
        mask = offsets < n_elements

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        add_f16 = (mm + residual).to(tl.float16)
        block_has_inf = tl.sum(
            tl.where((tl.abs(add_f16) == float("inf")) & mask, 1, 0),
            axis=0,
        )
        tl.atomic_or(flag_ptr, tl.where(block_has_inf > 0, 1, 0), sem="relaxed")

    @triton.jit
    def _t5_rmsnorm_scaled_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        flag_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        finite_bound: tl.constexpr,
        inf_bound: tl.constexpr,
        final_scale: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

        # aten.add on two fp16 tensors rounds before the explicit fp32 convert.
        add_f16 = (mm + residual).to(tl.float16)
        add_f32 = add_f16.to(tl.float32)

        has_inf = tl.load(flag_ptr) != 0
        bound = tl.where(has_inf, inf_bound, finite_bound)
        clamped_min = tl.where(add_f32 < -bound, -bound, add_f32)
        clamped = tl.where(clamped_min > bound, bound, clamped_min)
        clamped_f16 = clamped.to(tl.float16)
        x = clamped_f16.to(tl.float32)

        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        inv_rms = tl.rsqrt(sum_sq / hidden + eps)

        norm_f16 = (x * inv_rms).to(tl.float16)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        weighted_f16 = (weight * norm_f16).to(tl.float16)
        y = (weighted_f16 * final_scale).to(tl.float16)

        tl.store(out_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mm_95, residual, weight, shape0, shape1 = inputs
    tensor_inputs = (mm_95, residual, weight)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, RESIDUAL_SHAPE, WEIGHT_SHAPE)
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape0) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    return mm_95, residual, weight


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_t5_rmsnorm_scaled.py")

    mm_95, residual, weight = _validate_inputs(inputs)
    has_inf = torch.empty((1,), device=mm_95.device, dtype=torch.int32)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm_95.device,
        dtype=torch.float16,
    )

    _clear_flag_kernel[(1,)](has_inf)
    _any_inf_kernel[(triton.cdiv(TOTAL_ELEMENTS, BLOCK_ELEMS),)](
        mm_95,
        residual,
        has_inf,
        n_elements=TOTAL_ELEMENTS,
        block_elems=BLOCK_ELEMS,
    )
    _t5_rmsnorm_scaled_kernel[(ROWS,)](
        mm_95,
        residual,
        weight,
        has_inf,
        out,
        hidden=HIDDEN,
        eps=EPS,
        finite_bound=FINITE_BOUND,
        inf_bound=INF_BOUND,
        final_scale=FINAL_SCALE,
        block_h=BLOCK_H,
    )
    return out


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
