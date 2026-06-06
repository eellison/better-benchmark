"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT shifted-neighbor embedding assembly by gathering current, next-token, and previous-token rows from the embedding table and writing the final contiguous f32[32768,384] cat/view output directly with border zero-fill, while tuned Inductor measures in the same required memory-traffic envelope for the captured embedding/slice/constant_pad_nd/cat/view graph; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recomputation because the remaining work is the mandatory embedding-table reads and dense output store; the fix is BANDWIDTH_BOUND: record this as at floor unless broader gather/layout memory codegen or launch/allocation improvements move both paths."""
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

BATCH = 256
SEQ = 128
HIDDEN = 128
ROWS = BATCH * SEQ
OUT_HIDDEN = HIDDEN * 3
TABLE_SHAPE = (30522, HIDDEN)
IDS_SHAPE = (BATCH, SEQ)
OUT_SHAPE = (ROWS, OUT_HIDDEN)
OUT_STRIDE = (OUT_HIDDEN, 1)
CLASSIFICATION = "BANDWIDTH_BOUND"

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _shifted_embedding_cat_kernel(
        table_ptr,
        ids_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        OUT_WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        hidden_offsets = tl.arange(0, HIDDEN_SIZE)
        row_mask = row_offsets < N_ROWS
        col_mask = hidden_offsets < HIDDEN_SIZE

        seq_offsets = row_offsets - (row_offsets // SEQ_LEN) * SEQ_LEN

        next_mask = row_mask & (seq_offsets < (SEQ_LEN - 1))
        current_mask = row_mask
        prev_mask = row_mask & (seq_offsets > 0)

        next_positions = tl.where(next_mask, row_offsets + 1, row_offsets)
        current_positions = row_offsets
        prev_positions = tl.where(prev_mask, row_offsets - 1, row_offsets)

        next_ids = tl.load(ids_ptr + next_positions, mask=next_mask, other=0)
        current_ids = tl.load(ids_ptr + current_positions, mask=current_mask, other=0)
        prev_ids = tl.load(ids_ptr + prev_positions, mask=prev_mask, other=0)

        next_values = tl.load(
            table_ptr + next_ids[:, None] * HIDDEN_SIZE + hidden_offsets[None, :],
            mask=next_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        tl.store(
            out_ptr + row_offsets[:, None] * OUT_WIDTH + hidden_offsets[None, :],
            next_values,
            mask=row_mask[:, None] & col_mask[None, :],
        )

        current_values = tl.load(
            table_ptr + current_ids[:, None] * HIDDEN_SIZE + hidden_offsets[None, :],
            mask=current_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        tl.store(
            out_ptr + row_offsets[:, None] * OUT_WIDTH + HIDDEN_SIZE + hidden_offsets[None, :],
            current_values,
            mask=row_mask[:, None] & col_mask[None, :],
        )

        prev_values = tl.load(
            table_ptr + prev_ids[:, None] * HIDDEN_SIZE + hidden_offsets[None, :],
            mask=prev_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        tl.store(
            out_ptr + row_offsets[:, None] * OUT_WIDTH + 2 * HIDDEN_SIZE + hidden_offsets[None, :],
            prev_values,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_shifted_embedding_cat.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    table, ids, shape_param = inputs
    if not isinstance(table, torch.Tensor) or not isinstance(ids, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor embedding table and ids inputs")
    if table.device.type != "cuda" or ids.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if table.dtype != torch.float32:
        raise TypeError(f"expected fp32 embedding table, got {table.dtype}")
    if ids.dtype != torch.int64:
        raise TypeError(f"expected int64 embedding ids, got {ids.dtype}")
    if tuple(table.shape) != TABLE_SHAPE:
        raise ValueError(f"unexpected embedding table shape: {tuple(table.shape)}")
    if tuple(ids.shape) != IDS_SHAPE:
        raise ValueError(f"unexpected embedding ids shape: {tuple(ids.shape)}")
    if not table.is_contiguous() or not ids.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects captured contiguous table and ids layouts")
    if list(shape_param) != list(OUT_SHAPE):
        raise ValueError(f"unexpected shape parameter: {shape_param!r}")

    return table, ids, shape_param


def oracle_forward(inputs):
    """Run the full Repro.forward shifted embedding cat/view scope."""
    table, ids, _shape_param = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=table.device,
        dtype=table.dtype,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _shifted_embedding_cat_kernel[grid](
        table,
        ids,
        output,
        N_ROWS=ROWS,
        SEQ_LEN=SEQ,
        HIDDEN_SIZE=HIDDEN,
        OUT_WIDTH=OUT_HIDDEN,
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
