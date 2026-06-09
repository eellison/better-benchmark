"""
Oracle for sum_9ba8ffa3e0fb

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Computes the complete Longformer structured pool/upsample backward graph with a row-sum prepass and direct final-layout scatter/gather materialization.
  What Inductor change would fix: Recognize the structured Longformer slice-scatter/reduce/reshape pattern and lower it as one output-centric scatter-reduce producer with fused epilogues.
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
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


HEADS = 12
OVERLAP_BLOCKS = 4
TOKENS = 1024
CHUNK = 256
SOURCE_K = 768
REDUCE_K = 513
OUT_CHANNELS = 3
OUT_ROWS = 512
OUT_COLS = 512
FULL4_K = 257
FULL7_ROWS = 255
FULL7_COLS = 255

BLOCK_SUM_ROWS = 4
BLOCK_SUM_R = 1024
BLOCK_OUT = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _arg_offset(
        a,
        j,
        t,
        r,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
    ):
        b = a // HEADS_
        h = a - b * HEADS_
        s = j * CHUNK_ + t
        row = (b * TOKENS_ + s) * HEADS_ + h
        return row * REDUCE_K_ + r, row, b, h, s

    @triton.jit
    def _g_value(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        arg100_ptr,
        row_sums_ptr,
        full3_value,
        a,
        j,
        t,
        r,
        valid,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
    ):
        arg_off, row, b, _h, s = _arg_offset(
            a, j, t, r, HEADS_, TOKENS_, CHUNK_, REDUCE_K_
        )
        bmm_off = ((a * 4 + j) * CHUNK_ + t) * SOURCE_K_ + (t + r)

        softmax = tl.load(arg100_ptr + arg_off, mask=valid, other=0.0).to(tl.float32)
        row_sum = tl.load(row_sums_ptr + row, mask=valid, other=0.0).to(tl.float32)
        source = tl.load(bmm_ptr + bmm_off, mask=valid, other=0.0).to(tl.float32)
        keep = tl.load(arg101_ptr + arg_off, mask=valid, other=0) != 0
        force_full = tl.load(unsqueeze_ptr + b * TOKENS_ + s, mask=valid, other=0) != 0

        masked_source = tl.where(keep, source, 0.0)
        where_value = tl.where(force_full, full3_value, masked_source)
        mul_value = where_value * softmax
        return mul_value - softmax * row_sum

    @triton.jit
    def _full4_offset(
        b,
        t,
        h,
        u,
        HEADS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        FULL4_K_: tl.constexpr,
    ):
        return ((b * CHUNK_ + t) * HEADS_ + h) * FULL4_K_ + u

    @triton.jit
    def _full5_offset(
        b,
        t,
        h,
        r,
        HEADS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
    ):
        return ((b * CHUNK_ + t) * HEADS_ + h) * REDUCE_K_ + r

    @triton.jit
    def _full6_offset(
        b,
        s,
        h,
        r,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
    ):
        return ((b * TOKENS_ + s) * HEADS_ + h) * REDUCE_K_ + r

    @triton.jit
    def _a1_value(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        arg100_ptr,
        full4_ptr,
        convert3_ptr,
        full5_ptr,
        full6_ptr,
        row_sums_ptr,
        full3_value,
        a,
        j,
        t,
        r,
        valid,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL4_K_: tl.constexpr,
    ):
        g = _g_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            row_sums_ptr,
            full3_value,
            a,
            j,
            t,
            r,
            valid,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
        )
        arg_off, _row, b, h, s = _arg_offset(
            a, j, t, r, HEADS_, TOKENS_, CHUNK_, REDUCE_K_
        )
        full6 = tl.load(full6_ptr + arg_off, mask=valid, other=0.0).to(tl.float32)
        base = g + full6

        j3 = j == 3
        low = j3 & (r < 256)
        high = j3 & (r >= 256)
        full5 = tl.load(
            full5_ptr + _full5_offset(b, t, h, r, HEADS_, CHUNK_, REDUCE_K_),
            mask=valid & low,
            other=0.0,
        ).to(tl.float32)
        u = r - 256
        safe_u = tl.maximum(u, 0)
        full4 = tl.load(
            full4_ptr + _full4_offset(b, t, h, safe_u, HEADS_, CHUNK_, FULL4_K_),
            mask=valid & high,
            other=0.0,
        ).to(tl.float32)
        convert3 = tl.load(
            convert3_ptr + _full4_offset(b, t, h, safe_u, HEADS_, CHUNK_, FULL4_K_),
            mask=valid & high,
            other=0,
        ) != 0
        edge = tl.where(r >= 256, full4 + tl.where(convert3, full3_value, g), g + full5)
        return tl.where(j3, edge, base)

    @triton.jit
    def _a2_value(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        arg100_ptr,
        full4_ptr,
        convert3_ptr,
        full5_ptr,
        full6_ptr,
        convert4_ptr,
        row_sums_ptr,
        full3_value,
        a,
        j,
        t,
        r,
        valid,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL4_K_: tl.constexpr,
    ):
        a1 = _a1_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            row_sums_ptr,
            full3_value,
            a,
            j,
            t,
            r,
            valid,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
        )
        _arg_off, _row, b, h, s = _arg_offset(
            a, j, t, r, HEADS_, TOKENS_, CHUNK_, REDUCE_K_
        )
        full6 = tl.load(
            full6_ptr + _full6_offset(b, s, h, r, HEADS_, TOKENS_, REDUCE_K_),
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        nonzero_j = a1 + full6

        low = j == 0
        lower_cols = low & (r <= 256)
        full4 = tl.load(
            full4_ptr + _full4_offset(b, t, h, r, HEADS_, CHUNK_, FULL4_K_),
            mask=valid & lower_cols,
            other=0.0,
        ).to(tl.float32)
        convert4 = tl.load(
            convert4_ptr + _full4_offset(b, t, h, r, HEADS_, CHUNK_, FULL4_K_),
            mask=valid & lower_cols,
            other=0,
        ) != 0
        full5 = tl.load(
            full5_ptr + _full5_offset(b, t, h, r, HEADS_, CHUNK_, REDUCE_K_),
            mask=valid & low & (r > 256),
            other=0.0,
        ).to(tl.float32)
        j0 = tl.where(
            r <= 256,
            full4 + tl.where(convert4, full3_value, a1),
            a1 + full5,
        )
        return tl.where(low, j0, nonzero_j)

    @triton.jit
    def _b1_value(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        arg100_ptr,
        full4_ptr,
        convert3_ptr,
        full5_ptr,
        full6_ptr,
        convert4_ptr,
        full7_ptr,
        row_sums_ptr,
        full3_value,
        a,
        j,
        t,
        r,
        valid,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL4_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
    ):
        a2 = _a2_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            row_sums_ptr,
            full3_value,
            a,
            j,
            t,
            r,
            valid,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
        )
        replace = (j == 0) & (t > 0) & (r > 0) & (r < 256)
        full7 = tl.load(
            full7_ptr + (a * FULL7_ROWS_ + (t - 1)) * FULL7_COLS_ + (r - 1),
            mask=valid & replace,
            other=0.0,
        ).to(tl.float32)
        return tl.where(replace, full7, a2)

    @triton.jit
    def _b2_value(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        arg100_ptr,
        full4_ptr,
        convert3_ptr,
        full5_ptr,
        full6_ptr,
        convert4_ptr,
        full7_ptr,
        full11_ptr,
        row_sums_ptr,
        full3_value,
        a,
        j,
        t,
        r,
        valid,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL4_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
    ):
        b1 = _b1_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            full7_ptr,
            row_sums_ptr,
            full3_value,
            a,
            j,
            t,
            r,
            valid,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        replace = (j >= 1) & (r < 256)
        full11 = tl.load(
            full11_ptr + (((a * 3 + (j - 1)) * CHUNK_ + t) * 256 + r),
            mask=valid & replace,
            other=0.0,
        ).to(tl.float32)
        return tl.where(replace, full11, b1)

    @triton.jit
    def _b3_value(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        arg100_ptr,
        full4_ptr,
        convert3_ptr,
        full5_ptr,
        full6_ptr,
        convert4_ptr,
        full7_ptr,
        full11_ptr,
        full13_ptr,
        row_sums_ptr,
        full3_value,
        a,
        j,
        t,
        r,
        valid,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL4_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
    ):
        b2 = _b2_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            full7_ptr,
            full11_ptr,
            row_sums_ptr,
            full3_value,
            a,
            j,
            t,
            r,
            valid,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        replace = (j == 3) & (r >= 256)
        full13 = tl.load(
            full13_ptr + (a * CHUNK_ + t) * FULL4_K_ + (r - 256),
            mask=valid & replace,
            other=0.0,
        ).to(tl.float32)
        return tl.where(replace, full13, b2)

    @triton.jit
    def _row_sums_kernel(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        full3_ptr,
        arg100_ptr,
        row_sums_ptr,
        NUM_ROWS_: tl.constexpr,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_R_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        rr = tl.arange(0, BLOCK_R_)
        row = rows[None, :]
        r = rr[:, None]
        valid = (row < NUM_ROWS_) & (r < REDUCE_K_)

        h = row % HEADS_
        q = row // HEADS_
        s = q % TOKENS_
        b = q // TOKENS_
        a = b * HEADS_ + h
        j = s // CHUNK_
        t = s - j * CHUNK_

        arg_off = row * REDUCE_K_ + r
        bmm_off = ((a * 4 + j) * CHUNK_ + t) * SOURCE_K_ + (t + r)

        full3 = tl.load(full3_ptr).to(tl.float32)
        softmax = tl.load(arg100_ptr + arg_off, mask=valid, other=0.0).to(tl.float32)
        source = tl.load(bmm_ptr + bmm_off, mask=valid, other=0.0).to(tl.float32)
        keep = tl.load(arg101_ptr + arg_off, mask=valid, other=0) != 0
        force_full = tl.load(unsqueeze_ptr + b * TOKENS_ + s, mask=valid, other=0) != 0
        where_value = tl.where(force_full, full3, tl.where(keep, source, 0.0))
        values = where_value * softmax
        sums = tl.sum(tl.where(valid, values, 0.0), axis=0)
        tl.store(row_sums_ptr + rows, sums, mask=rows < NUM_ROWS_)

    @triton.jit
    def _materialize_a2_kernel(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        full3_ptr,
        arg100_ptr,
        full4_ptr,
        convert3_ptr,
        full5_ptr,
        full6_ptr,
        convert4_ptr,
        row_sums_ptr,
        a2_ptr,
        TOTAL_A2_: tl.constexpr,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL4_K_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        valid = offsets < TOTAL_A2_

        r = offsets % REDUCE_K_
        q = offsets // REDUCE_K_
        t = q % CHUNK_
        q = q // CHUNK_
        j = q % 4
        a = q // 4

        full3 = tl.load(full3_ptr).to(tl.float32)
        value = _a2_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            row_sums_ptr,
            full3,
            a,
            j,
            t,
            r,
            valid,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
        )
        tl.store(a2_ptr + offsets, value, mask=valid)

    @triton.jit
    def _a2_buffer_value(
        a2_ptr,
        a,
        j,
        t,
        r,
        valid,
        CHUNK_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
    ):
        offsets = (((a * 4 + j) * CHUNK_ + t) * REDUCE_K_ + r)
        return tl.load(a2_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    @triton.jit
    def _b1_buffer_value(
        a2_ptr,
        full7_ptr,
        a,
        j,
        t,
        r,
        valid,
        CHUNK_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
    ):
        a2 = _a2_buffer_value(a2_ptr, a, j, t, r, valid, CHUNK_, REDUCE_K_)
        replace = (j == 0) & (t > 0) & (r > 0) & (r < 256)
        full7 = tl.load(
            full7_ptr + (a * FULL7_ROWS_ + (t - 1)) * FULL7_COLS_ + (r - 1),
            mask=valid & replace,
            other=0.0,
        ).to(tl.float32)
        return tl.where(replace, full7, a2)

    @triton.jit
    def _b2_buffer_value(
        a2_ptr,
        full7_ptr,
        full11_ptr,
        a,
        j,
        t,
        r,
        valid,
        CHUNK_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
    ):
        b1 = _b1_buffer_value(
            a2_ptr,
            full7_ptr,
            a,
            j,
            t,
            r,
            valid,
            CHUNK_,
            REDUCE_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        replace = (j >= 1) & (r < 256)
        full11 = tl.load(
            full11_ptr + (((a * 3 + (j - 1)) * CHUNK_ + t) * 256 + r),
            mask=valid & replace,
            other=0.0,
        ).to(tl.float32)
        return tl.where(replace, full11, b1)

    @triton.jit
    def _b3_buffer_value(
        a2_ptr,
        full7_ptr,
        full11_ptr,
        full13_ptr,
        a,
        j,
        t,
        r,
        valid,
        CHUNK_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        FULL4_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
    ):
        b2 = _b2_buffer_value(
            a2_ptr,
            full7_ptr,
            full11_ptr,
            a,
            j,
            t,
            r,
            valid,
            CHUNK_,
            REDUCE_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        replace = (j == 3) & (r >= 256)
        full13 = tl.load(
            full13_ptr + (a * CHUNK_ + t) * FULL4_K_ + (r - 256),
            mask=valid & replace,
            other=0.0,
        ).to(tl.float32)
        return tl.where(replace, full13, b2)

    @triton.jit
    def _final_output_from_a2_kernel(
        a2_ptr,
        full7_ptr,
        full8_ptr,
        full9_ptr,
        full10_ptr,
        full11_ptr,
        full12_ptr,
        full13_ptr,
        full14_ptr,
        out_ptr,
        TOTAL_: tl.constexpr,
        CHUNK_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        OUT_ROWS_: tl.constexpr,
        OUT_COLS_: tl.constexpr,
        OUT_CHANNELS_: tl.constexpr,
        FULL4_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        valid = offsets < TOTAL_

        p = offsets // (OUT_ROWS_ * OUT_COLS_)
        rem = offsets - p * OUT_ROWS_ * OUT_COLS_
        v = rem // OUT_COLS_
        w = rem - v * OUT_COLS_

        add_linear = (p * (OUT_ROWS_ + 1) + v) * OUT_COLS_ + w
        q = add_linear // REDUCE_K_
        col = add_linear - q * REDUCE_K_
        row = q % OUT_ROWS_
        pc = q // OUT_ROWS_
        c = pc % OUT_CHANNELS_
        a = pc // OUT_CHANNELS_

        full10_off = (((a * OUT_CHANNELS_ + c) * OUT_ROWS_ + row) * REDUCE_K_ + col)
        full10 = tl.load(full10_ptr + full10_off, mask=valid, other=0.0).to(tl.float32)

        c0_a2_mask = valid & (c == 0) & (row < 255) & (col >= 258)
        c0_a2 = _a2_buffer_value(
            a2_ptr,
            a,
            0,
            row + 1,
            col - 257,
            c0_a2_mask,
            CHUNK_,
            REDUCE_K_,
        )
        full8 = tl.load(
            full8_ptr + (a * FULL7_ROWS_ + row) * REDUCE_K_ + col,
            mask=valid & (c == 0) & (row < 255) & (col < 258),
            other=0.0,
        ).to(tl.float32)
        full9 = tl.load(
            full9_ptr + (a * OUT_ROWS_ + row) * REDUCE_K_ + col,
            mask=valid & (c == 0) & (row >= 255),
            other=0.0,
        ).to(tl.float32)
        c0_edge = tl.where(row < 255, tl.where(col >= 258, c0_a2, full8), full9)
        term0 = tl.where(c == 0, c0_edge, full10)

        d1_mid = (row >= 255) & (row < 511)
        d1_b1_mask = valid & d1_mid & (col >= 257)
        d1_b1 = _b1_buffer_value(
            a2_ptr,
            full7_ptr,
            a,
            c + 1,
            row - 255,
            col - 257,
            d1_b1_mask,
            CHUNK_,
            REDUCE_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        d1_full12 = tl.load(
            full12_ptr + (((a * OUT_CHANNELS_ + c) * CHUNK_ + (row - 255)) * REDUCE_K_ + col),
            mask=valid & d1_mid & (col < 257),
            other=0.0,
        ).to(tl.float32)
        term1 = tl.where(d1_mid, tl.where(col >= 257, d1_b1, d1_full12), full10)

        c2_row = row >= 256
        c2_b2_mask = valid & (c == 2) & c2_row & (col <= 256)
        c2_b2 = _b2_buffer_value(
            a2_ptr,
            full7_ptr,
            full11_ptr,
            a,
            3,
            row - 256,
            col + 256,
            c2_b2_mask,
            CHUNK_,
            REDUCE_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        c2_full14 = tl.load(
            full14_ptr + (a * CHUNK_ + (row - 256)) * REDUCE_K_ + col,
            mask=valid & (c == 2) & c2_row & (col > 256),
            other=0.0,
        ).to(tl.float32)
        c2_full9 = tl.load(
            full9_ptr + (a * OUT_ROWS_ + row) * REDUCE_K_ + col,
            mask=valid & (c == 2) & (~c2_row),
            other=0.0,
        ).to(tl.float32)
        c2_edge = tl.where(c2_row, tl.where(col <= 256, c2_b2, c2_full14), c2_full9)
        term2 = tl.where(c == 2, c2_edge, full10)

        c3_row = row < 256
        c3_b3_mask = valid & c3_row & (col <= 256)
        c3_b3 = _b3_buffer_value(
            a2_ptr,
            full7_ptr,
            full11_ptr,
            full13_ptr,
            a,
            c,
            row,
            col + 256,
            c3_b3_mask,
            CHUNK_,
            REDUCE_K_,
            FULL4_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        c3_full12 = tl.load(
            full12_ptr + (((a * OUT_CHANNELS_ + c) * CHUNK_ + row) * REDUCE_K_ + col),
            mask=valid & c3_row & (col > 256),
            other=0.0,
        ).to(tl.float32)
        term3 = tl.where(c3_row, tl.where(col <= 256, c3_b3, c3_full12), full10)

        tl.store(out_ptr + offsets, term0 + term1 + term2 + term3, mask=valid)

    @triton.jit
    def _final_output_kernel(
        bmm_ptr,
        arg101_ptr,
        unsqueeze_ptr,
        full3_ptr,
        arg100_ptr,
        full4_ptr,
        convert3_ptr,
        full5_ptr,
        full6_ptr,
        convert4_ptr,
        full7_ptr,
        full8_ptr,
        full9_ptr,
        full10_ptr,
        full11_ptr,
        full12_ptr,
        full13_ptr,
        full14_ptr,
        row_sums_ptr,
        out_ptr,
        TOTAL_: tl.constexpr,
        HEADS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        CHUNK_: tl.constexpr,
        SOURCE_K_: tl.constexpr,
        REDUCE_K_: tl.constexpr,
        OUT_ROWS_: tl.constexpr,
        OUT_COLS_: tl.constexpr,
        OUT_CHANNELS_: tl.constexpr,
        FULL4_K_: tl.constexpr,
        FULL7_ROWS_: tl.constexpr,
        FULL7_COLS_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        valid = offsets < TOTAL_

        p = offsets // (OUT_ROWS_ * OUT_COLS_)
        rem = offsets - p * OUT_ROWS_ * OUT_COLS_
        v = rem // OUT_COLS_
        w = rem - v * OUT_COLS_

        add_linear = (p * (OUT_ROWS_ + 1) + v) * OUT_COLS_ + w
        q = add_linear // REDUCE_K_
        col = add_linear - q * REDUCE_K_
        row = q % OUT_ROWS_
        pc = q // OUT_ROWS_
        c = pc % OUT_CHANNELS_
        a = pc // OUT_CHANNELS_

        full3 = tl.load(full3_ptr).to(tl.float32)
        full10_off = (((a * OUT_CHANNELS_ + c) * OUT_ROWS_ + row) * REDUCE_K_ + col)
        full10 = tl.load(full10_ptr + full10_off, mask=valid, other=0.0).to(tl.float32)

        # select_scatter_default_1 contribution
        c0_a2_mask = valid & (c == 0) & (row < 255) & (col >= 258)
        c0_a2 = _a2_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            row_sums_ptr,
            full3,
            a,
            0,
            row + 1,
            col - 257,
            c0_a2_mask,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
        )
        full8 = tl.load(
            full8_ptr + (a * FULL7_ROWS_ + row) * REDUCE_K_ + col,
            mask=valid & (c == 0) & (row < 255) & (col < 258),
            other=0.0,
        ).to(tl.float32)
        full9 = tl.load(
            full9_ptr + (a * OUT_ROWS_ + row) * REDUCE_K_ + col,
            mask=valid & (c == 0) & (row >= 255),
            other=0.0,
        ).to(tl.float32)
        c0_edge = tl.where(row < 255, tl.where(col >= 258, c0_a2, full8), full9)
        term0 = tl.where(c == 0, c0_edge, full10)

        # slice_scatter_default_17 contribution
        d1_mid = (row >= 255) & (row < 511)
        d1_b1_mask = valid & d1_mid & (col >= 257)
        d1_b1 = _b1_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            full7_ptr,
            row_sums_ptr,
            full3,
            a,
            c + 1,
            row - 255,
            col - 257,
            d1_b1_mask,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        d1_full12 = tl.load(
            full12_ptr + (((a * OUT_CHANNELS_ + c) * CHUNK_ + (row - 255)) * REDUCE_K_ + col),
            mask=valid & d1_mid & (col < 257),
            other=0.0,
        ).to(tl.float32)
        term1 = tl.where(d1_mid, tl.where(col >= 257, d1_b1, d1_full12), full10)

        # select_scatter_default_3 contribution
        c2_row = row >= 256
        c2_b2_mask = valid & (c == 2) & c2_row & (col <= 256)
        c2_b2 = _b2_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            full7_ptr,
            full11_ptr,
            row_sums_ptr,
            full3,
            a,
            3,
            row - 256,
            col + 256,
            c2_b2_mask,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        c2_full14 = tl.load(
            full14_ptr + (a * CHUNK_ + (row - 256)) * REDUCE_K_ + col,
            mask=valid & (c == 2) & c2_row & (col > 256),
            other=0.0,
        ).to(tl.float32)
        c2_full9 = tl.load(
            full9_ptr + (a * OUT_ROWS_ + row) * REDUCE_K_ + col,
            mask=valid & (c == 2) & (~c2_row),
            other=0.0,
        ).to(tl.float32)
        c2_edge = tl.where(c2_row, tl.where(col <= 256, c2_b2, c2_full14), c2_full9)
        term2 = tl.where(c == 2, c2_edge, full10)

        # slice_scatter_default_22 contribution
        c3_row = row < 256
        c3_b3_mask = valid & c3_row & (col <= 256)
        c3_b3 = _b3_value(
            bmm_ptr,
            arg101_ptr,
            unsqueeze_ptr,
            arg100_ptr,
            full4_ptr,
            convert3_ptr,
            full5_ptr,
            full6_ptr,
            convert4_ptr,
            full7_ptr,
            full11_ptr,
            full13_ptr,
            row_sums_ptr,
            full3,
            a,
            c,
            row,
            col + 256,
            c3_b3_mask,
            HEADS_,
            TOKENS_,
            CHUNK_,
            SOURCE_K_,
            REDUCE_K_,
            FULL4_K_,
            FULL7_ROWS_,
            FULL7_COLS_,
        )
        c3_full12 = tl.load(
            full12_ptr + (((a * OUT_CHANNELS_ + c) * CHUNK_ + row) * REDUCE_K_ + col),
            mask=valid & c3_row & (col > 256),
            other=0.0,
        ).to(tl.float32)
        term3 = tl.where(c3_row, tl.where(col <= 256, c3_b3, c3_full12), full10)

        value = term0 + term1 + term2 + term3
        tl.store(out_ptr + offsets, value, mask=valid)


def _check_expected_shapes(inputs):
    bmm_45 = inputs[0]
    arg101_1 = inputs[3]
    unsqueeze_1 = inputs[4]
    full_3 = inputs[5]
    arg100_1 = inputs[6]
    full_4 = inputs[7]
    convert_element_type_3 = inputs[8]
    full_5 = inputs[9]
    full_6 = inputs[10]
    convert_element_type_4 = inputs[11]

    if bmm_45.ndim != 3 or bmm_45.shape[1:] != (CHUNK, SOURCE_K):
        raise ValueError(f"unexpected bmm_45 shape: {tuple(bmm_45.shape)}")
    if bmm_45.shape[0] % OVERLAP_BLOCKS != 0:
        raise ValueError(f"unexpected bmm_45 leading dimension: {bmm_45.shape[0]}")

    a_count = bmm_45.shape[0] // OVERLAP_BLOCKS
    batch = a_count // HEADS
    expected = {
        "arg101_1": (batch, TOKENS, HEADS, REDUCE_K),
        "unsqueeze_1": (batch, TOKENS, 1, 1),
        "arg100_1": (batch, TOKENS, HEADS, REDUCE_K),
        "full_4": (batch, CHUNK, HEADS, FULL4_K),
        "convert_element_type_3": (batch, CHUNK, HEADS, FULL4_K),
        "full_5": (batch, CHUNK, HEADS, REDUCE_K),
        "full_6": (batch, TOKENS, HEADS, REDUCE_K),
        "convert_element_type_4": (batch, CHUNK, HEADS, FULL4_K),
    }
    actual = {
        "arg101_1": tuple(arg101_1.shape),
        "unsqueeze_1": tuple(unsqueeze_1.shape),
        "arg100_1": tuple(arg100_1.shape),
        "full_4": tuple(full_4.shape),
        "convert_element_type_3": tuple(convert_element_type_3.shape),
        "full_5": tuple(full_5.shape),
        "full_6": tuple(full_6.shape),
        "convert_element_type_4": tuple(convert_element_type_4.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"unexpected {name} shape: {actual[name]} != {shape}")
    if a_count != batch * HEADS:
        raise ValueError(f"unexpected batch/head factoring: a_count={a_count}")
    if full_3.numel() != 1:
        raise ValueError(f"unexpected full_3 shape: {tuple(full_3.shape)}")
    return a_count, batch


@oracle_impl(hardware="H100", shapes="(T([384, 256, 768], f32), T([96, 4, 256, 769], f32), T([96, 4, 197120], f32), T([8, 1024, 12, 513], b8), T([8, 1024, 1, 1], b8), T([], f32), T([8, 1024, 12, 513], f32), T([8, 256, 12, 257], f32), T([8, 256, 12, 257], b8), T([8, 256, 12, 513], f32), T([8, 1024, 12, 513], f32), T([8, 256, 12, 257], b8), T([96, 255, 255], f32), T([96, 255, 513], f32), T([96, 512, 513], f32), T([96, 3, 512, 513], f32), T([96, 3, 256, 256], f32), T([96, 3, 256, 513], f32), T([96, 256, 257], f32), T([96, 256, 513], f32), S([96, 4, 256, 768, 1]), S([96, 4, 196864]), S([96, 4, 256, 770]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([96, 4, 256, 513]), S([96, 3, 513, 512]), S([96, 3, 512, 512, 1]), S([288, 512, 512]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    (
        bmm_45,
        _full,
        _full_1,
        arg101_1,
        unsqueeze_1,
        full_3,
        arg100_1,
        full_4,
        convert_element_type_3,
        full_5,
        full_6,
        convert_element_type_4,
        full_7,
        full_8,
        full_9,
        full_10,
        full_11,
        full_12,
        full_13,
        full_14,
        *_shape_params,
    ) = inputs

    if bmm_45.device.type != "cuda" or triton is None:
        return get_repro_instance()(*inputs)

    a_count, batch = _check_expected_shapes(inputs)
    if not all(
        tensor.is_contiguous()
        for tensor in (
            bmm_45,
            arg101_1,
            unsqueeze_1,
            full_3,
            arg100_1,
            full_4,
            convert_element_type_3,
            full_5,
            full_6,
            convert_element_type_4,
            full_7,
            full_8,
            full_9,
            full_10,
            full_11,
            full_12,
            full_13,
            full_14,
        )
    ):
        raise ValueError("oracle expects the captured contiguous input layouts")

    num_reduce_rows = batch * TOKENS * HEADS
    row_sums = torch.empty((num_reduce_rows,), device=bmm_45.device, dtype=torch.float32)
    _row_sums_kernel[(triton.cdiv(num_reduce_rows, BLOCK_SUM_ROWS),)](
        bmm_45,
        arg101_1,
        unsqueeze_1,
        full_3,
        arg100_1,
        row_sums,
        NUM_ROWS_=num_reduce_rows,
        HEADS_=HEADS,
        TOKENS_=TOKENS,
        CHUNK_=CHUNK,
        SOURCE_K_=SOURCE_K,
        REDUCE_K_=REDUCE_K,
        BLOCK_ROWS_=BLOCK_SUM_ROWS,
        BLOCK_R_=BLOCK_SUM_R,
        num_warps=8,
    )

    a2 = torch.empty(
        (a_count, OVERLAP_BLOCKS, CHUNK, REDUCE_K),
        device=bmm_45.device,
        dtype=torch.float32,
    )
    _materialize_a2_kernel[(triton.cdiv(a2.numel(), BLOCK_OUT),)](
        bmm_45,
        arg101_1,
        unsqueeze_1,
        full_3,
        arg100_1,
        full_4,
        convert_element_type_3,
        full_5,
        full_6,
        convert_element_type_4,
        row_sums,
        a2,
        TOTAL_A2_=a2.numel(),
        HEADS_=HEADS,
        TOKENS_=TOKENS,
        CHUNK_=CHUNK,
        SOURCE_K_=SOURCE_K,
        REDUCE_K_=REDUCE_K,
        FULL4_K_=FULL4_K,
        BLOCK_=BLOCK_OUT,
        num_warps=4,
    )

    out = torch.empty(
        (a_count * OUT_CHANNELS, OUT_ROWS, OUT_COLS),
        device=bmm_45.device,
        dtype=torch.float32,
    )
    total = out.numel()
    _final_output_from_a2_kernel[(triton.cdiv(total, BLOCK_OUT),)](
        a2,
        full_7,
        full_8,
        full_9,
        full_10,
        full_11,
        full_12,
        full_13,
        full_14,
        out,
        TOTAL_=total,
        CHUNK_=CHUNK,
        REDUCE_K_=REDUCE_K,
        OUT_ROWS_=OUT_ROWS,
        OUT_COLS_=OUT_COLS,
        OUT_CHANNELS_=OUT_CHANNELS,
        FULL4_K_=FULL4_K,
        FULL7_ROWS_=FULL7_ROWS,
        FULL7_COLS_=FULL7_COLS,
        BLOCK_=BLOCK_OUT,
        num_warps=4,
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
