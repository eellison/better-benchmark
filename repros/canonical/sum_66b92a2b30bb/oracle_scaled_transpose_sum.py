"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Whisper reshape/multiply scope by streaming the contiguous `[12000, 384]` backing layout once, writing the returned `[384, 12000]` transpose-view buffer and accumulating the `[384]` column sum from the same scaled producer, whereas Inductor currently schedules the scale/materialized transpose side output and the small column reduction as separate generic kernels around the view chain; Inductor cannot do this today because its scheduler/codegen does not form a coordinated multi-output reduction plan when a pointwise producer has both a required layout side output and a tiny compatible reduction output; the fix is SCHEDULER_FUSION: add a multi-output reduction template that stores the side-output backing layout and atomically accumulates the sibling column sum from shared row-tile partials."""
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
ROWS = 8 * 1500
HIDDEN = 6 * 64
INPUT_SHAPE = (8, 6, 1500, 64)
INPUT_STRIDE = (576000, 64, 384, 1)
VIEW0_SHAPE = (8, 1500, 384)
VIEW1_SHAPE = (12000, 384)
SUM_SHAPE = (384,)
SCALE = 0.125
BLOCK_M = 256
BLOCK_N = 32

if triton is not None:

    @triton.jit
    def _scaled_transpose_sum_atomic_kernel(
        input_ptr,
        layout_base_ptr,
        sum_out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        cols = pid_n * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        offsets = rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)

        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        values = values * SCALE_

        tl.store(layout_base_ptr + offsets, values, mask=mask)
        partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
        tl.atomic_add(sum_out_ptr + cols, partial, sem="relaxed", mask=cols < N)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    getitem, shape0, shape1, shape2 = inputs
    getitem_t = _expect_tensor(
        "getitem",
        getitem,
        INPUT_SHAPE,
        INPUT_STRIDE,
        torch.float32,
    )
    if _shape_tuple(shape0) != VIEW0_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != VIEW1_SHAPE:
        raise ValueError(f"unexpected second view shape parameter: {shape1!r}")
    if _shape_tuple(shape2) != SUM_SHAPE:
        raise ValueError(f"unexpected final view shape parameter: {shape2!r}")
    return getitem_t


def _num_m_blocks() -> int:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_scaled_transpose_sum.py")
    return triton.cdiv(ROWS, BLOCK_M)


def oracle_forward(inputs):
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
        raise RuntimeError("Triton is required for oracle_scaled_transpose_sum.py")

    getitem = _validate_inputs(inputs)
    num_m_blocks = _num_m_blocks()

    layout_base = torch.empty((ROWS, HIDDEN), device=getitem.device, dtype=torch.float32)
    sum_out = torch.empty((HIDDEN,), device=getitem.device, dtype=torch.float32)
    sum_out.zero_()

    _scaled_transpose_sum_atomic_kernel[
        (num_m_blocks, triton.cdiv(HIDDEN, BLOCK_N))
    ](
        getitem,
        layout_base,
        sum_out,
        M=ROWS,
        N=HIDDEN,
        SCALE_=SCALE,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
        num_stages=1,
    )

    return layout_base.permute(1, 0), sum_out


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
