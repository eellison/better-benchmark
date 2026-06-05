"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the required divided broadcast backing tensor for the returned `[256, 12000]` transpose while also producing the sibling `[256]` sum directly from the original `[8, 256]` source rows in one Triton kernel, whereas Inductor currently schedules the expand/divide/view/permute materialization and reduction as generic separate consumers of the same logical producer; Inductor cannot do this today because its scheduler/codegen does not form a multi-output materializing reduction that fuses a required layout-changing side output with a compatible broadcasted reduction and simplifies the fixed repeated-row accumulator; the fix is SCHEDULER_FUSION: teach Inductor to fuse view/expand/divide producers into a shared materialization-plus-reduction schedule that emits the transpose backing store and direct source-row reduction together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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


BATCH = 8
REPEAT = 1500
COLS = 256
TOTAL_ROWS = BATCH * REPEAT
BASE_SHAPE = (TOTAL_ROWS, COLS)
TRANSPOSE_SHAPE = (COLS, TOTAL_ROWS)
TRANSPOSE_STRIDE = (1, COLS)
SUM_SHAPE = (COLS,)
SUM_STRIDE = (1,)
BLOCK_ROWS = 64
BLOCK_COLS = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _materialize_and_sum_kernel(
        mm_ptr,
        base_out_ptr,
        sum_out_ptr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_COLS_: tl.constexpr,
        BATCH_: tl.constexpr,
        REPEAT_: tl.constexpr,
        COLS_: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        repeat_tile = tl.program_id(1)
        col_tile = tl.program_id(2)

        repeat_offsets = repeat_tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        cols = col_tile * BLOCK_COLS_ + tl.arange(0, BLOCK_COLS_)
        row_offsets = batch_id * REPEAT_ + repeat_offsets

        col_mask = cols < COLS_
        store_mask = (repeat_offsets[:, None] < REPEAT_) & col_mask[None, :]

        mm_values = tl.load(
            mm_ptr + batch_id * COLS_ + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        values = tl.zeros((BLOCK_ROWS_, BLOCK_COLS_), dtype=tl.float32) + (
            mm_values[None, :] / REPEAT_
        )
        tl.store(
            base_out_ptr + row_offsets[:, None] * COLS_ + cols[None, :],
            values,
            mask=store_mask,
        )

        sum_rows = tl.arange(0, BATCH_)
        sum_values = tl.load(
            mm_ptr + sum_rows[:, None] * COLS_ + cols[None, :],
            mask=col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(sum_values, axis=0)
        sum_mask = (batch_id == 0) & (repeat_tile == 0) & col_mask
        tl.store(sum_out_ptr + cols, sums, mask=sum_mask)


def _shape_tuple(value):
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    mm, shape_param_0, shape_param_1, shape_param_2 = inputs
    if not isinstance(mm, torch.Tensor):
        raise TypeError(f"expected input 0 to be a tensor, got {type(mm)!r}")
    if mm.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if mm.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {mm.dtype}")
    if tuple(mm.shape) != (BATCH, COLS):
        raise ValueError(f"unexpected mm shape: {tuple(mm.shape)}")
    if tuple(mm.stride()) != (COLS, 1) or not mm.is_contiguous():
        raise ValueError(f"unexpected mm stride: {tuple(mm.stride())}")
    if _shape_tuple(shape_param_0) != (BATCH, REPEAT, COLS):
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != BASE_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    if _shape_tuple(shape_param_2) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2!r}")
    return mm


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
    mm = _validate_inputs(inputs)
    base = torch.empty(BASE_SHAPE, device=mm.device, dtype=mm.dtype)
    summed = torch.empty_strided(SUM_SHAPE, SUM_STRIDE, device=mm.device, dtype=mm.dtype)

    grid = (
        BATCH,
        triton.cdiv(REPEAT, BLOCK_ROWS),
        triton.cdiv(COLS, BLOCK_COLS),
    )
    _materialize_and_sum_kernel[grid](
        mm,
        base,
        summed,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_COLS_=BLOCK_COLS,
        BATCH_=BATCH,
        REPEAT_=REPEAT,
        COLS_=COLS,
        num_warps=8,
        num_stages=4,
    )
    transposed = base.permute(1, 0)
    if tuple(transposed.shape) != TRANSPOSE_SHAPE or transposed.stride() != TRANSPOSE_STRIDE:
        raise RuntimeError("oracle produced an unexpected transpose layout")
    return transposed, summed


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
