"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GPT-2 QKV bf16 permute/view/cat scope by writing the returned contiguous `[B*S, 2304]` concat and accumulating row-tile partial column sums from the same traversal, then finalizing the f32 reduction with the captured bf16-to-f32 rounding boundary, whereas Inductor schedules the live returned cat materialization and sibling sum as generic layout/reduction work over the concatenated tensor; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K plan that fuses a required materialized concat output with compatible column partial reductions across multiple strided inputs while preserving output scope and casts; the fix is COOPERATIVE_SPLIT_K: add a multi-input layout-materialization plus split-row reduction template that emits the returned cat buffer and finalized rounded reduction together."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


QKV_CHUNK = 768
OUT_COLS = 2304


@triton.jit
def _cat_partial_sum_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cat_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    QKV: tl.constexpr,
    OUT: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    source = tl.program_id(2)
    rows = row_tile * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    inner_cols = col_tile * COL_BLOCK + tl.arange(0, COL_BLOCK)
    out_cols = source * QKV + inner_cols
    row_mask = rows < ROWS
    col_mask = inner_cols < QKV
    mask = row_mask[:, None] & col_mask[None, :]

    source_offsets = rows[:, None] * QKV + inner_cols[None, :]
    src_ptr = tl.where(source == 0, q_ptr, v_ptr)
    src_ptr = tl.where(source == 2, k_ptr, src_ptr)
    values = tl.load(src_ptr + source_offsets, mask=mask, other=0.0)

    cat_offsets = rows[:, None] * OUT + out_cols[None, :]
    tl.store(cat_ptr + cat_offsets, values, mask=mask)

    values_f32 = values.to(tl.float32)
    partial = tl.sum(tl.where(mask, values_f32, 0.0), axis=0)
    tl.store(partial_ptr + row_tile * OUT + out_cols, partial, mask=col_mask)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    out_ptr,
    NUM_ROW_TILES: tl.constexpr,
    OUT: tl.constexpr,
    PARTIAL_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    row_tiles = tl.arange(0, PARTIAL_BLOCK)
    mask = (row_tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < OUT)
    values = tl.load(
        partial_ptr + row_tiles[:, None] * OUT + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + cols, rounded, mask=cols < OUT)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 1a3ec418: (T([32,12,512,64], bf16) x3, S([32,512,768]) x3, S([16384,2304]), S([2304]))
@oracle_impl(
    hardware="B200",
    point="1a3ec418",
    ROW_BLOCK=64,
    COL_BLOCK=32,
    FINAL_COL_BLOCK=32,
    num_warps=8,
)
# ce8e60c0: (T([8,12,1024,64], bf16) x3, S([8,1024,768]) x3, S([8192,2304]), S([2304]))
@oracle_impl(
    hardware="B200",
    point="ce8e60c0",
    ROW_BLOCK=64,
    COL_BLOCK=32,
    FINAL_COL_BLOCK=32,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    COL_BLOCK: int,
    FINAL_COL_BLOCK: int,
    num_warps: int,
):
    q, k, v, _shape0, _shape1, _shape2, cat_shape, sum_shape = inputs
    rows, cols = _shape_tuple(cat_shape)
    cat = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=q.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        _shape_tuple(sum_shape),
        (1,),
        device=q.device,
        dtype=torch.float32,
    )

    num_row_tiles = triton.cdiv(rows, ROW_BLOCK)
    partial = torch.empty_strided(
        (num_row_tiles, cols),
        (cols, 1),
        device=q.device,
        dtype=torch.float32,
    )

    _cat_partial_sum_kernel[(num_row_tiles, triton.cdiv(QKV_CHUNK, COL_BLOCK), 3)](
        q,
        k,
        v,
        cat,
        partial,
        ROWS=rows,
        QKV=QKV_CHUNK,
        OUT=OUT_COLS,
        ROW_BLOCK=ROW_BLOCK,
        COL_BLOCK=COL_BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    _final_sum_kernel[(triton.cdiv(cols, FINAL_COL_BLOCK),)](
        partial,
        out,
        NUM_ROW_TILES=num_row_tiles,
        OUT=OUT_COLS,
        PARTIAL_BLOCK=triton.next_power_of_2(num_row_tiles),
        COL_BLOCK=FINAL_COL_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    return cat, out
