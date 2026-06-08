"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full exact-erf/exp pointwise materialization and accumulates row-block column partials in that same Triton pass before the final partial reduction, whereas Inductor materializes the pointwise tensor and launches a separate reduction that rereads it; Inductor cannot do this today because its scheduler does not coordinate a live materialized side output with split row partial accumulators for the sibling sum consumer; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to emit pointwise materialization plus grouped partial accumulation when the producer is also returned as a view."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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


ROWS = 25216
COLS = 3072
VIEW3_SHAPE = (128, 197, 3072)
MATRIX_SHAPE = (ROWS, COLS)
SUM_SHAPE = (COLS,)
ROW_BLOCK = 32
NUM_GROUPS = (ROWS + ROW_BLOCK - 1) // ROW_BLOCK
FINAL_GROUP_BLOCK = 1024
FINAL_BLOCK_N = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None and libdevice is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=4),
        ],
        key=["rows", "cols"],
    )
    @triton.jit
    def _materialize_partial_kernel(
        mm_ptr,
        addmm_ptr,
        base_ptr,
        partial_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        row_block: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        col_block = tl.program_id(1)

        row_offsets = tl.arange(0, row_block)[:, None]
        col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        row_indices = group_id * row_block + row_offsets
        valid = (row_indices < rows) & (col_offsets < cols)
        offsets = row_indices * cols + col_offsets

        tmp0 = tl.load(mm_ptr + offsets, valid, other=0.0).to(tl.float32)
        tmp1 = tl.load(addmm_ptr + offsets, valid, other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], 0.7071067811865476, tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = libdevice.erf(tmp3)
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.full([1, 1], 0.5, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp1 * tmp1
        tmp10 = tl.full([1, 1], -0.5, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = libdevice.exp(tmp11)
        tmp13 = tl.full([1, 1], 0.3989422804014327, tl.float32)
        tmp14 = tmp12 * tmp13
        tmp15 = tmp1 * tmp14
        tmp16 = tmp8 + tmp15
        tmp17 = tmp0 * tmp16

        tl.store(base_ptr + offsets, tmp17, valid)
        partial = tl.sum(tl.where(valid, tmp17, 0.0), axis=0)
        partial_cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            partial_ptr + group_id * cols + partial_cols,
            partial,
            partial_cols < cols,
        )

    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        out_ptr,
        cols: tl.constexpr,
        num_groups: tl.constexpr,
        group_block: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        group_offsets = tl.arange(0, group_block)[:, None]
        col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        valid = (group_offsets < num_groups) & (col_offsets < cols)
        values = tl.load(
            partial_ptr + group_offsets * cols + col_offsets,
            valid,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        totals = tl.sum(tl.where(valid, values, 0.0), axis=0)
        out_cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(out_ptr + out_cols, totals, out_cols < cols)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(inputs):
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_90, addmm_2, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(mm_90, torch.Tensor) or not isinstance(addmm_2, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects two tensor inputs followed by shape parameters")
    for name, value in (("mm_90", mm_90), ("addmm_2", addmm_2)):
        if value.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be f32, got {value.dtype}")
        if tuple(value.shape) != MATRIX_SHAPE:
            raise ValueError(f"{name} shape must be {MATRIX_SHAPE}, got {tuple(value.shape)}")
        if tuple(value.stride()) != (COLS, 1):
            raise ValueError(f"{name} stride must be {(COLS, 1)}, got {tuple(value.stride())}")

    shape0_t = _shape_tuple(shape0)
    shape1_t = _shape_tuple(shape1)
    shape2_t = _shape_tuple(shape2)
    shape3_t = _shape_tuple(shape3)
    if shape0_t != VIEW3_SHAPE or shape1_t != VIEW3_SHAPE:
        raise ValueError(f"captured 3D view shapes must be {VIEW3_SHAPE}")
    if shape2_t != MATRIX_SHAPE:
        raise ValueError(f"captured matrix view shape must be {MATRIX_SHAPE}")
    if shape3_t != SUM_SHAPE:
        raise ValueError(f"captured sum view shape must be {SUM_SHAPE}")

    return mm_90, addmm_2, shape2_t, shape3_t


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
    mm_90, addmm_2, matrix_shape, sum_shape = _validate_inputs(inputs)
    rows, cols = matrix_shape
    num_groups = (rows + ROW_BLOCK - 1) // ROW_BLOCK
    if num_groups != NUM_GROUPS:
        raise ValueError(f"captured row grouping must produce {NUM_GROUPS} groups")

    base = torch.empty_strided(matrix_shape, (cols, 1), device=mm_90.device, dtype=torch.float32)
    partial = torch.empty_strided(
        (num_groups, cols),
        (cols, 1),
        device=mm_90.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(sum_shape, (1,), device=mm_90.device, dtype=torch.float32)

    grid = lambda meta: (num_groups, triton.cdiv(cols, meta["BLOCK_N"]))
    _materialize_partial_kernel[grid](
        mm_90,
        addmm_2,
        base,
        partial,
        rows,
        cols,
        ROW_BLOCK,
    )
    _final_sum_kernel[(triton.cdiv(cols, FINAL_BLOCK_N),)](
        partial,
        sum_out,
        cols,
        num_groups,
        FINAL_GROUP_BLOCK,
        FINAL_BLOCK_N,
    )

    return (base.permute(1, 0), sum_out.reshape(sum_shape))


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
