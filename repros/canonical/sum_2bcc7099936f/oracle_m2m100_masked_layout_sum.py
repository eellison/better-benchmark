"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete M2M100 masked materialization plus sibling column sum scope returned by Repro.forward, including the f32[8192,4096] to [64,128,4096] view, bool where(mask, 0.0, mm), returned [4096,8192] permute view over the materialized backing storage, and reshaped f32[4096] column sum, by streaming the shared masked producer once into Triton partial reductions; Inductor currently schedules the layout materialization and the reduction as generic work over the decomposed view/where/view/permute/sum chain; Inductor cannot do this today because its scheduler does not fuse a dense materialized side output with a compatible sibling reduction over the same masked producer while preserving the returned strided view contract; the fix is SCHEDULER_FUSION: add a full-scope pointwise-materialization-plus-column-reduction schedule that writes the required backing storage and reduction partials from one producer traversal."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
from oracle_harness import (  # noqa: E402
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
ROWS = 8192
COLS = 4096
VIEW_SHAPE = (64, 128, COLS)
INPUT_SHAPE = (ROWS, COLS)
INPUT_STRIDE = (COLS, 1)
MASK_SHAPE = VIEW_SHAPE
MASK_STRIDE = (128 * COLS, COLS, 1)
BACKING_SHAPE = (ROWS, COLS)
BACKING_STRIDE = (COLS, 1)
PERMUTE_SHAPE = (COLS, ROWS)
PERMUTE_STRIDE = (1, COLS)
SUM_BASE_SHAPE = (1, COLS)
SUM_BASE_STRIDE = (COLS, 1)
SUM_SHAPE = (COLS,)
SUM_STRIDE = (1,)

BLOCK_ROWS = 128
BLOCK_COLS = 32
FINAL_BLOCK_COLS = 32


if triton is not None:

    @triton.jit
    def _masked_materialize_partial_sum_kernel(
        mm_ptr,
        mask_ptr,
        backing_ptr,
        partials_ptr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_COLS_: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)

        rows = row_tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        cols = col_tile * BLOCK_COLS_ + tl.arange(0, BLOCK_COLS_)
        active = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)
        offsets = rows[:, None] * COLS_ + cols[None, :]

        mm_values = tl.load(mm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mask_values = tl.load(mask_ptr + offsets, mask=active, other=1)
        values = tl.where(mask_values, 0.0, mm_values)
        values = tl.where(active, values, 0.0)

        tl.store(backing_ptr + offsets, values, mask=active)
        partials = tl.sum(values, axis=0)
        tl.store(
            partials_ptr + row_tile * COLS_ + cols,
            partials,
            mask=cols < COLS_,
        )

    @triton.jit
    def _finish_column_sum_kernel(
        partials_ptr,
        sum_base_ptr,
        NUM_ROW_TILES_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_COLS_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_COLS_ + tl.arange(0, BLOCK_COLS_)
        tiles = tl.arange(0, BLOCK_TILES_)
        active = (tiles[:, None] < NUM_ROW_TILES_) & (cols[None, :] < COLS_)
        values = tl.load(
            partials_ptr + tiles[:, None] * COLS_ + cols[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(tl.where(active, values, 0.0), axis=0)
        tl.store(sum_base_ptr + cols, sums, mask=cols < COLS_)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


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
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mm, arg39_1, shape_param_0, shape_param_1, shape_param_2 = inputs
    mm = _expect_tensor("mm", mm, INPUT_SHAPE, INPUT_STRIDE, torch.float32)
    arg39_1 = _expect_tensor("arg39_1", arg39_1, MASK_SHAPE, MASK_STRIDE, torch.bool)
    if arg39_1.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if _shape_tuple(shape_param_0) != VIEW_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != INPUT_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    if _shape_tuple(shape_param_2) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2!r}")
    return mm, arg39_1


def oracle_forward(inputs):
    """Run the exact full-scope masked materialization plus column sum."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_m2m100_masked_layout_sum.py")

    mm, arg39_1 = _validate_inputs(inputs)
    backing = torch.empty_strided(
        BACKING_SHAPE,
        BACKING_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    sum_base = torch.empty_strided(
        SUM_BASE_SHAPE,
        SUM_BASE_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    num_row_tiles = triton.cdiv(ROWS, BLOCK_ROWS)
    partials = torch.empty_strided(
        (num_row_tiles, COLS),
        (COLS, 1),
        device=mm.device,
        dtype=torch.float32,
    )

    grid = (num_row_tiles, triton.cdiv(COLS, BLOCK_COLS))
    _masked_materialize_partial_sum_kernel[grid](
        mm,
        arg39_1,
        backing,
        partials,
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_COLS_=BLOCK_COLS,
        num_warps=8,
        num_stages=4,
    )
    _finish_column_sum_kernel[(triton.cdiv(COLS, FINAL_BLOCK_COLS),)](
        partials,
        sum_base,
        NUM_ROW_TILES_=num_row_tiles,
        COLS_=COLS,
        BLOCK_TILES_=triton.next_power_of_2(num_row_tiles),
        BLOCK_COLS_=FINAL_BLOCK_COLS,
        num_warps=4,
        num_stages=4,
    )

    permute_out = torch.as_strided(backing, PERMUTE_SHAPE, PERMUTE_STRIDE)
    sum_out = torch.as_strided(sum_base, SUM_SHAPE, SUM_STRIDE)
    return permute_out, sum_out


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
