"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Blenderbot backward tail from Repro.forward by streaming the shared `mm_6 + mm_8 + mm_10` producer into row-local hidden-dimension partials, column batch/sequence partials, and the returned `[32, 128, 2560]` epilogue tensor while preserving the two `[2560]` reduction outputs, whereas Inductor currently schedules the hidden-dimension reductions, batch/sequence reductions, and dependent epilogue as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen cannot form one coordinated cross-axis multi-output reduction with row partials, column partials, and recomputed pointwise epilogue consumers; the fix is COOPERATIVE_SPLIT_K: add a split-K multi-output reduction template that emits shared producer loads, row/column partial finalization, and the dependent epilogue in one planned schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_ed8809aa30cf"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 2560
INV_CHANNELS = 1.0 / CHANNELS

def get_inputs():
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _reduction_partials_kernel(
    mm6_ptr,
    mm8_ptr,
    mm10_ptr,
    weight_ptr,
    source_ptr,
    row_shift_ptr,
    row_scale_ptr,
    row_partial0_ptr,
    row_partial1_ptr,
    col_partial0_ptr,
    col_partial1_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    NUM_C_BLOCKS_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS_
    col_mask = cols < CHANNELS_
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * CHANNELS_ + cols[None, :]

    mm6 = tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm8 = tl.load(mm8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm10 = tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = mm6 + mm8 + mm10
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    m2 = (source - row_shift[:, None]) * row_scale[:, None]
    scaled = x * weight[None, :]

    row_sum0 = tl.sum(tl.where(mask, scaled, 0.0), axis=1)
    row_sum1 = tl.sum(tl.where(mask, scaled * m2, 0.0), axis=1)
    row_partial_offsets = rows * NUM_C_BLOCKS_ + col_block
    tl.store(row_partial0_ptr + row_partial_offsets, row_sum0, mask=row_mask)
    tl.store(row_partial1_ptr + row_partial_offsets, row_sum1, mask=row_mask)

    col_sum0 = tl.sum(tl.where(mask, x * m2, 0.0), axis=0)
    col_sum1 = tl.sum(tl.where(mask, x, 0.0), axis=0)
    col_partial_offsets = row_block * CHANNELS_ + cols
    tl.store(col_partial0_ptr + col_partial_offsets, col_sum0, mask=col_mask)
    tl.store(col_partial1_ptr + col_partial_offsets, col_sum1, mask=col_mask)


@triton.jit
def _finalize_row_sums_kernel(
    row_partial0_ptr,
    row_partial1_ptr,
    row_sum0_ptr,
    row_sum1_ptr,
    ROWS_: tl.constexpr,
    NUM_C_BLOCKS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    tiles = tl.arange(0, BLOCK_TILES)
    row_mask = rows < ROWS_
    tile_mask = tiles < NUM_C_BLOCKS_
    mask = row_mask[:, None] & tile_mask[None, :]
    offsets = rows[:, None] * NUM_C_BLOCKS_ + tiles[None, :]

    partial0 = tl.load(row_partial0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial1 = tl.load(row_partial1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(row_sum0_ptr + rows, tl.sum(partial0, axis=1), mask=row_mask)
    tl.store(row_sum1_ptr + rows, tl.sum(partial1, axis=1), mask=row_mask)


@triton.jit
def _finalize_col_sums_kernel(
    col_partial0_ptr,
    col_partial1_ptr,
    out0_ptr,
    out1_ptr,
    NUM_ROW_BLOCKS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    col_mask = cols < CHANNELS_
    block_mask = blocks < NUM_ROW_BLOCKS_
    mask = block_mask[:, None] & col_mask[None, :]
    offsets = blocks[:, None] * CHANNELS_ + cols[None, :]

    partial0 = tl.load(col_partial0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial1 = tl.load(col_partial1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out0_ptr + cols, tl.sum(partial0, axis=0), mask=col_mask)
    tl.store(out1_ptr + cols, tl.sum(partial1, axis=0), mask=col_mask)


@triton.jit
def _epilogue_kernel(
    mm6_ptr,
    mm8_ptr,
    mm10_ptr,
    weight_ptr,
    source_ptr,
    row_shift_ptr,
    row_scale_ptr,
    residual_ptr,
    row_sum0_ptr,
    row_sum1_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_CHANNELS_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_
    cols = offsets % CHANNELS_
    rows = offsets // CHANNELS_

    mm6 = tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm8 = tl.load(mm8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm10 = tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = mm6 + mm8 + mm10
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum0 = tl.load(row_sum0_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    row_sum1 = tl.load(row_sum1_ptr + rows, mask=mask, other=0.0).to(tl.float32)

    scaled = x * weight
    m2 = (source - row_shift) * row_scale
    ln_tail = row_scale * INV_CHANNELS_ * (scaled * CHANNELS_ - row_sum0 - m2 * row_sum1)
    tl.store(out_ptr + offsets, residual + ln_tail, mask=mask)

def oracle_fused(
    mm_6: torch.Tensor,
    mm_8: torch.Tensor,
    mm_10: torch.Tensor,
    arg0_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg9_1: torch.Tensor,
    arg10_1: torch.Tensor,
    add_3: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_6.shape == (ROWS, CHANNELS)
    assert mm_8.shape == (ROWS, CHANNELS)
    assert mm_10.shape == (ROWS, CHANNELS)
    assert arg0_1.shape == (CHANNELS,)
    assert arg1_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg9_1.shape == (BATCH, SEQ, 1)
    assert arg10_1.shape == (BATCH, SEQ, 1)
    assert add_3.shape == (BATCH, SEQ, CHANNELS)
    assert mm_6.is_contiguous()
    assert mm_8.is_contiguous()
    assert mm_10.is_contiguous()
    assert arg0_1.is_contiguous()
    assert arg1_1.is_contiguous()
    assert arg9_1.is_contiguous()
    assert arg10_1.is_contiguous()
    assert add_3.is_contiguous()

    device = mm_6.device
    block_r = 32
    block_c = 64
    num_row_blocks = triton.cdiv(ROWS, block_r)
    num_c_blocks = triton.cdiv(CHANNELS, block_c)

    row_partials = torch.empty((2, ROWS, num_c_blocks), device=device, dtype=torch.float32)
    col_partials = torch.empty((2, num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    row_sums = torch.empty((2, ROWS), device=device, dtype=torch.float32)
    vector_outputs = torch.empty((2, CHANNELS), device=device, dtype=torch.float32)
    out_tensor = torch.empty((BATCH, SEQ, CHANNELS), device=device, dtype=torch.float32)

    _reduction_partials_kernel[(num_row_blocks, num_c_blocks)](
        mm_6,
        mm_8,
        mm_10,
        arg0_1,
        arg1_1,
        arg9_1,
        arg10_1,
        row_partials[0],
        row_partials[1],
        col_partials[0],
        col_partials[1],
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        NUM_C_BLOCKS_=num_c_blocks,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=4,
    )

    row_block_rows = 8
    _finalize_row_sums_kernel[(triton.cdiv(ROWS, row_block_rows),)](
        row_partials[0],
        row_partials[1],
        row_sums[0],
        row_sums[1],
        ROWS_=ROWS,
        NUM_C_BLOCKS_=num_c_blocks,
        BLOCK_ROWS=row_block_rows,
        BLOCK_TILES=triton.next_power_of_2(num_c_blocks),
        num_warps=4,
    )

    finalize_block_c = 16
    _finalize_col_sums_kernel[(triton.cdiv(CHANNELS, finalize_block_c),)](
        col_partials[0],
        col_partials[1],
        vector_outputs[0],
        vector_outputs[1],
        NUM_ROW_BLOCKS_=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=finalize_block_c,
        num_warps=8,
    )

    numel = ROWS * CHANNELS
    block_elems = 1024
    _epilogue_kernel[(triton.cdiv(numel, block_elems),)](
        mm_6,
        mm_8,
        mm_10,
        arg0_1,
        arg1_1,
        arg9_1,
        arg10_1,
        add_3,
        row_sums[0],
        row_sums[1],
        out_tensor,
        NUMEL_=numel,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return vector_outputs[0], vector_outputs[1], out_tensor

def oracle_forward(inputs):
    return oracle_fused(*inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
