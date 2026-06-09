"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the full masked f32 matrix backing the returned transpose view and computes the sibling column sum from the same loaded `where(arg14_1 <= 0, full, mm_7)` values in one shape-specific Triton tile, whereas Inductor currently lowers this multi-output pointwise-plus-reduction scope through its generic fused scheduler with broader indexing and reduction structure; Inductor cannot do this today because the scheduler/codegen does not select a specialized multi-output reduction template that shares a pointwise producer between a required layout-changing side output and a compatible dim-0 sum consumer; the fix is SCHEDULER_FUSION: teach Inductor to fuse shared pointwise producers into a materialize-plus-reduce schedule that writes the transpose backing storage and column reduction together."""
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


SUPPORTED_SHAPES = {(2048, 512), (256, 1024)}


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _masked_materialize_sum_kernel(
        arg_ptr,
        full_ptr,
        mm_ptr,
        transpose_storage_ptr,
        sum_ptr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_ROWS)
        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

        row_col_offsets = rows[:, None] * COLS + cols[None, :]
        mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)

        arg = tl.load(arg_ptr + row_col_offsets, mask=mask, other=1.0).to(tl.float32)
        mm = tl.load(mm_ptr + row_col_offsets, mask=mask, other=0.0).to(tl.float32)
        fill = tl.load(full_ptr).to(tl.float32)
        values = tl.where(arg <= 0.0, fill, mm)

        tl.store(transpose_storage_ptr + row_col_offsets, values, mask=mask)
        sums = tl.sum(tl.where(mask, values, 0.0), axis=0)
        tl.store(sum_ptr + cols, sums, mask=cols < COLS)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    arg14_1, full, mm_7, shape_param_0 = inputs
    if not isinstance(arg14_1, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(arg14_1)!r}")
    if not isinstance(full, torch.Tensor):
        raise TypeError(f"expected tensor input 1, got {type(full)!r}")
    if not isinstance(mm_7, torch.Tensor):
        raise TypeError(f"expected tensor input 2, got {type(mm_7)!r}")
    if arg14_1.device.type != "cuda" or mm_7.device.type != "cuda" or full.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if arg14_1.dtype != torch.float32 or mm_7.dtype != torch.float32 or full.dtype != torch.float32:
        raise TypeError(
            f"expected float32 inputs, got {arg14_1.dtype}, {full.dtype}, {mm_7.dtype}"
        )
    if full.dim() != 0:
        raise ValueError(f"expected scalar full tensor, got shape {tuple(full.shape)}")
    if tuple(arg14_1.shape) != tuple(mm_7.shape):
        raise ValueError(f"arg14_1/mm_7 shape mismatch: {tuple(arg14_1.shape)} vs {tuple(mm_7.shape)}")
    if not arg14_1.is_contiguous() or not mm_7.is_contiguous():
        raise ValueError(f"expected contiguous matrix inputs, got strides {arg14_1.stride()} and {mm_7.stride()}")

    rows, cols = (int(arg14_1.shape[0]), int(arg14_1.shape[1]))
    if (rows, cols) not in SUPPORTED_SHAPES:
        raise ValueError(f"unexpected matrix shape for {REPRO_ID}: {(rows, cols)}")
    if _shape_tuple(shape_param_0) != (cols,):
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    return arg14_1, full, mm_7, rows, cols


def _kernel_config(rows: int) -> tuple[int, int, int]:
    block_rows = triton.next_power_of_2(rows)
    if rows >= 1024:
        return int(block_rows), 4, 8
    return int(block_rows), 16, 8


@oracle_impl(hardware="H100", shapes="(T([2048, 512], f32), T([], f32), T([2048, 512], f32), S([512]))")
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
    arg14_1, full, mm_7, rows, cols = _validate_inputs(inputs)
    permuted = torch.empty_strided(
        (cols, rows),
        (1, cols),
        device=arg14_1.device,
        dtype=arg14_1.dtype,
    )
    summed = torch.empty_strided((cols,), (1,), device=arg14_1.device, dtype=arg14_1.dtype)

    block_rows, block_cols, num_warps = _kernel_config(rows)
    grid = (triton.cdiv(cols, block_cols),)
    _masked_materialize_sum_kernel[grid](
        arg14_1,
        full,
        mm_7,
        permuted,
        summed,
        ROWS=rows,
        COLS=cols,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=num_warps,
        num_stages=3,
    )
    return permuted, summed


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
