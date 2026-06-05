"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete int64-to-int32, add-zero, masked multiply, int64 add-one, negative-index wrapping, bounds check, embedding gather, and final [64,128,1024] view as a row-tiled Triton gather that hoists each computed row index across the 1024 hidden columns, whereas Inductor currently emits one generic pointwise indirect-index kernel over all 8,388,608 output elements that recomputes flat div/mod, reloads the same cumsum and mask values for every hidden column, and performs per-element indirect-index checks; Inductor cannot do this today because its scheduler/codegen pattern library does not recognize computed-index embedding gathers as a structured row gather with column-vectorized loads and hoisted index arithmetic; the fix is NEW_PATTERN: add a computed-index embedding-gather lowering that groups contiguous hidden columns by token row, hoists the index expression and guard once per row, and emits direct dense table loads into the final layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - py_compile should not require Triton.
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS0 = 64
ROWS1 = 128
TOKENS = ROWS0 * ROWS1
VOCAB = 1026
HIDDEN = 1024
OUT_SHAPE = (ROWS0, ROWS1, HIDDEN)
OUT_STRIDE = (ROWS1 * HIDDEN, HIDDEN, 1)

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _computed_embedding_gather_kernel(
        cumsum_ptr,
        mask_ptr,
        table_ptr,
        out_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = tl.arange(0, BLOCK_N)
        token_mask = token_offsets < 8192
        col_mask = col_offsets < 1024

        cumsum_i32 = tl.load(cumsum_ptr + token_offsets, mask=token_mask, other=0).to(tl.int32)
        mask_i32 = tl.load(mask_ptr + token_offsets, mask=token_mask, other=0)
        row = (cumsum_i32 * mask_i32).to(tl.int64) + 1
        wrapped_row = tl.where(row < 0, row + 1026, row)
        tl.device_assert(
            (0 <= wrapped_row) & (wrapped_row < 1026),
            "index out of bounds: 0 <= wrapped_row < 1026",
        )

        offsets = wrapped_row[:, None] * 1024 + col_offsets[None, :]
        values = tl.load(
            table_ptr + offsets,
            mask=token_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        out_offsets = token_offsets[:, None] * 1024 + col_offsets[None, :]
        tl.store(
            out_ptr + out_offsets,
            values,
            mask=token_mask[:, None] & col_mask[None, :],
        )


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_computed_embedding_gather.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    cumsum, mask, table, shape_param = inputs
    if not isinstance(cumsum, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor cumsum and mask inputs")
    if not isinstance(table, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects a tensor embedding table")
    if cumsum.device.type != "cuda" or mask.device.type != "cuda" or table.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if cumsum.dtype != torch.int64:
        raise TypeError(f"expected int64 cumsum, got {cumsum.dtype}")
    if mask.dtype != torch.int32:
        raise TypeError(f"expected int32 mask, got {mask.dtype}")
    if table.dtype != torch.float32:
        raise TypeError(f"expected fp32 embedding table, got {table.dtype}")
    if tuple(cumsum.shape) != (ROWS0, ROWS1):
        raise ValueError(f"unexpected cumsum shape: {tuple(cumsum.shape)}")
    if tuple(mask.shape) != (ROWS0, ROWS1):
        raise ValueError(f"unexpected mask shape: {tuple(mask.shape)}")
    if tuple(table.shape) != (VOCAB, HIDDEN):
        raise ValueError(f"unexpected embedding table shape: {tuple(table.shape)}")
    if not cumsum.is_contiguous() or not mask.is_contiguous() or not table.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects captured contiguous input layouts")
    if list(shape_param) != list(OUT_SHAPE):
        raise ValueError(f"unexpected output shape parameter: {shape_param!r}")

    return cumsum, mask, table, shape_param


def oracle_forward(inputs):
    """Run the full Repro.forward scope with a row-specialized computed gather."""
    cumsum, mask, table, shape_param = _validate_inputs(inputs)
    output = torch.empty_strided(
        tuple(shape_param),
        OUT_STRIDE,
        device=table.device,
        dtype=table.dtype,
    )
    grid = lambda meta: (triton.cdiv(TOKENS, meta["BLOCK_M"]),)
    _computed_embedding_gather_kernel[grid](
        cumsum,
        mask,
        table,
        output,
    )
    return output


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
