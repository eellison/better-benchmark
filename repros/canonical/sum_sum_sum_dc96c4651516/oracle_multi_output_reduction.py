"""
Oracle for sum_sum_sum_dc96c4651516

Gap diagnosis:
  Classification: COOPERATIVE_SPLIT_K
  What oracle does differently: Streams the Swin window-unpartition/index producer once per row tile, computes the row-local channel reductions, writes the materialized transpose backing buffer, and accumulates the three returned channel reductions as cooperative partials.
  What Inductor change would fix: Add cooperative split-K multi-output reduction support for producers that must both materialize a side output and return small sibling reductions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 128
WINDOW_GRID = 8
WINDOW = 7
HEIGHT = WINDOW_GRID * WINDOW
WIDTH = HEIGHT
CHANNELS = 128
WINDOW_AREA = WINDOW * WINDOW
HW = HEIGHT * WIDTH
ROWS = BATCH * HW

if tl is not None:
    TL_WINDOW_GRID = tl.constexpr(WINDOW_GRID)
    TL_WINDOW = tl.constexpr(WINDOW)
    TL_WIDTH = tl.constexpr(WIDTH)
    TL_CHANNELS = tl.constexpr(CHANNELS)
    TL_WINDOW_AREA = tl.constexpr(WINDOW_AREA)
    TL_HW = tl.constexpr(HW)
    TL_ROWS = tl.constexpr(ROWS)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _store_and_partial_reduce_kernel(
        mm_ptr,
        index_ptr,
        weight_ptr,
        rhs_ptr,
        gate_ptr,
        residual_ptr,
        out_base_ptr,
        partials_ptr,
        NUM_ROW_BLOCKS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        c = tl.arange(0, BLOCK_C)

        row_mask = rows < TL_ROWS
        c_mask = c < TL_CHANNELS
        mask = row_mask[:, None] & c_mask[None, :]

        n = rows // TL_HW
        hw = rows - n * TL_HW
        h = hw // TL_WIDTH
        w = hw - h * TL_WIDTH

        indexed_h = tl.load(index_ptr + h, mask=row_mask, other=0)
        indexed_w = tl.load(index_ptr + w, mask=row_mask, other=0)
        block_h = indexed_h // TL_WINDOW
        block_w = indexed_w // TL_WINDOW
        inner_h = indexed_h - block_h * TL_WINDOW
        inner_w = indexed_w - block_w * TL_WINDOW

        window_row = ((n * TL_WINDOW_GRID + block_h) * TL_WINDOW_GRID + block_w) * TL_WINDOW_AREA
        source_row = window_row + inner_h * TL_WINDOW + inner_w

        source_offsets = source_row[:, None] * TL_CHANNELS + c[None, :]
        dense_offsets = rows[:, None] * TL_CHANNELS + c[None, :]

        source = tl.load(mm_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        weighted = source * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        value = residual + gate[:, None] * (
            weighted * TL_CHANNELS - row_sum[:, None] - rhs * row_dot[:, None]
        )

        tl.store(out_base_ptr + dense_offsets, value, mask=mask)

        partial0 = tl.sum(tl.where(mask, source * rhs, 0.0), axis=0)
        partial1 = tl.sum(tl.where(mask, source, 0.0), axis=0)
        partial3 = tl.sum(tl.where(mask, value, 0.0), axis=0)

        partial_offsets = row_block * TL_CHANNELS + c
        plane = NUM_ROW_BLOCKS * TL_CHANNELS
        tl.store(partials_ptr + partial_offsets, partial0, mask=c_mask)
        tl.store(partials_ptr + plane + partial_offsets, partial1, mask=c_mask)
        tl.store(partials_ptr + plane * 2 + partial_offsets, partial3, mask=c_mask)


    @triton.jit
    def _reduce_partials_stage1_kernel(
        partials_ptr,
        chunk_partials_ptr,
        NUM_ROW_BLOCKS: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        GROUP_BLOCKS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        blocks = chunk * GROUP_BLOCKS + tl.arange(0, GROUP_BLOCKS)

        mask = (blocks[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < TL_CHANNELS)
        offsets = blocks[:, None] * TL_CHANNELS + c[None, :]
        in_plane = NUM_ROW_BLOCKS * TL_CHANNELS
        out_offsets = chunk * TL_CHANNELS + c
        out_plane = NUM_CHUNKS * TL_CHANNELS

        vals0 = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        vals1 = tl.load(partials_ptr + in_plane + offsets, mask=mask, other=0.0).to(tl.float32)
        vals3 = tl.load(partials_ptr + in_plane * 2 + offsets, mask=mask, other=0.0).to(tl.float32)

        c_mask = c < TL_CHANNELS
        tl.store(chunk_partials_ptr + out_offsets, tl.sum(vals0, axis=0), mask=c_mask)
        tl.store(chunk_partials_ptr + out_plane + out_offsets, tl.sum(vals1, axis=0), mask=c_mask)
        tl.store(chunk_partials_ptr + out_plane * 2 + out_offsets, tl.sum(vals3, axis=0), mask=c_mask)


    @triton.jit
    def _reduce_partials_stage2_kernel(
        chunk_partials_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        chunks = tl.arange(0, BLOCK_CHUNKS)

        mask = (chunks[:, None] < NUM_CHUNKS) & (c[None, :] < TL_CHANNELS)
        offsets = chunks[:, None] * TL_CHANNELS + c[None, :]
        plane = NUM_CHUNKS * TL_CHANNELS

        vals0 = tl.load(chunk_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        vals1 = tl.load(chunk_partials_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32)
        vals3 = tl.load(chunk_partials_ptr + plane * 2 + offsets, mask=mask, other=0.0).to(tl.float32)

        c_mask = c < TL_CHANNELS
        tl.store(out0_ptr + c, tl.sum(vals0, axis=0), mask=c_mask)
        tl.store(out1_ptr + c, tl.sum(vals1, axis=0), mask=c_mask)
        tl.store(out3_ptr + c, tl.sum(vals3, axis=0), mask=c_mask)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same four outputs: two channel reductions, the materialized transposed side
    output, and the final channel reduction over that side output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        mm_190,
        arg196_1,
        arg11_1,
        arg192_1,
        arg554_1,
        view_750,
        *_shape_params,
    ) = inputs

    device = mm_190.device
    out0 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out3 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    block_rows = 128
    group_blocks = 256
    reduce_block_c = 32
    num_row_blocks = triton.cdiv(ROWS, block_rows)
    num_chunks = triton.cdiv(num_row_blocks, group_blocks)
    block_chunks = triton.next_power_of_2(num_chunks)

    partials = torch.empty((3, num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    chunk_partials = torch.empty((3, num_chunks, CHANNELS), device=device, dtype=torch.float32)

    _store_and_partial_reduce_kernel[(num_row_blocks,)](
        mm_190,
        arg196_1,
        arg11_1,
        arg192_1,
        arg554_1,
        view_750,
        out_base,
        partials,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_ROWS=block_rows,
        BLOCK_C=CHANNELS,
        num_warps=8,
    )

    _reduce_partials_stage1_kernel[(num_chunks, triton.cdiv(CHANNELS, reduce_block_c))](
        partials,
        chunk_partials,
        NUM_ROW_BLOCKS=num_row_blocks,
        NUM_CHUNKS=num_chunks,
        GROUP_BLOCKS=group_blocks,
        BLOCK_C=reduce_block_c,
        num_warps=8,
    )

    _reduce_partials_stage2_kernel[(triton.cdiv(CHANNELS, reduce_block_c),)](
        chunk_partials,
        out0,
        out1,
        out3,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=reduce_block_c,
        num_warps=4,
    )

    return out0, out1, out_base.permute(1, 0), out3


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
