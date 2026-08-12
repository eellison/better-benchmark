"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 BERT/LayoutLM attention layout clone scope, writing the returned contiguous `[B*S, 768]` tensor and its `[768, B*S]` transpose view while accumulating the sibling fp32 column sum from the same source traversal with the required bf16 rounding before the returned fp32 sum, whereas Inductor lowers the layout materialization and reduction as separate scheduled work over the cloned buffer; Inductor cannot do this today because its scheduler does not fuse a materialized layout-copy producer with a sibling reduction output that consumes the same producer elements while preserving aliasing and cast boundaries; the fix is SCHEDULER_FUSION: add a multi-output attention-layout copy plus reduction schedule that emits the clone backing, metadata-only transpose view, and rounded reduction result from one producer traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _copy_partial_sum_kernel(
    input_ptr,
    base_ptr,
    partial_ptr,
    XNUMEL: tl.constexpr,
    COLS: tl.constexpr,
    SEQ: tl.constexpr,
    XBLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    seq = tl.arange(0, SBLOCK)
    mask = (x[:, None] < XNUMEL) & (seq[None, :] < SEQ)

    batch = x // COLS
    col = x - batch * COLS
    values = tl.load(input_ptr + x[:, None] * SEQ + seq[None, :], mask=mask, other=0.0).to(tl.float32)
    base_offsets = batch[:, None] * SEQ * COLS + seq[None, :] * COLS + col[:, None]
    tl.store(base_ptr + base_offsets, values, mask=mask)

    partial = tl.sum(tl.where(mask, values, 0.0), axis=1)
    tl.store(partial_ptr + x, partial, mask=x < XNUMEL)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    out_ptr,
    BATCH: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    batches = tl.arange(0, BLOCK_B)
    mask = (batches[:, None] < BATCH) & (cols[None, :] < COLS)
    values = tl.load(partial_ptr + batches[:, None] * COLS + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + cols, rounded, mask=cols < COLS)


@triton.jit
def _row_tile_copy_partial_sum_kernel(
    input_ptr,
    base_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = rows // SEQ
    seq = rows - batch * SEQ
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)

    input_offsets = (batch[:, None] * COLS + cols[None, :]) * SEQ + seq[:, None]
    base_offsets = rows[:, None] * COLS + cols[None, :]
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(base_ptr + base_offsets, values, mask=mask)

    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(partial_ptr + row_tile * COLS + cols, partial, mask=cols < COLS)


@triton.jit
def _final_row_tile_sum_kernel(
    partial_ptr,
    out_ptr,
    NUM_ROW_TILES: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_tiles = tl.arange(0, BLOCK_R)
    mask = (row_tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < COLS)
    values = tl.load(
        partial_ptr + row_tiles[:, None] * COLS + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + cols, rounded, mask=cols < COLS)


def _shape(value):
    return tuple(int(dim) for dim in value)


@oracle_impl(hardware="B200", point="1e7ad64a", XBLOCK=32, SBLOCK=128, BLOCK_B=16, BLOCK_C=32, num_warps=8)
@oracle_impl(
    hardware="B200",
    point="17865e49",
    XBLOCK=8,
    SBLOCK=512,
    BLOCK_B=32,
    BLOCK_C=32,
    ROW_BLOCK=128,
    COL_BLOCK=16,
    PARTIAL_BLOCK=128,
    FINAL_BLOCK_C=16,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    XBLOCK: int,
    SBLOCK: int,
    BLOCK_B: int,
    BLOCK_C: int,
    num_warps: int,
    ROW_BLOCK: int = 0,
    COL_BLOCK: int = 0,
    PARTIAL_BLOCK: int = 0,
    FINAL_BLOCK_C: int = 0,
):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    base_shape = _shape(_shape_param_2)
    sum_shape = _shape(_shape_param_3)
    rows, cols = base_shape
    seq = int(arg0_1.shape[2])
    batch = rows // seq

    base = torch.empty(base_shape, device=arg0_1.device, dtype=torch.bfloat16)
    out_sum = torch.empty(sum_shape, device=arg0_1.device, dtype=torch.float32)

    if ROW_BLOCK:
        num_row_tiles = triton.cdiv(rows, ROW_BLOCK)
        partial = torch.empty((num_row_tiles, cols), device=arg0_1.device, dtype=torch.float32)
        _row_tile_copy_partial_sum_kernel[(num_row_tiles, triton.cdiv(cols, COL_BLOCK))](
            arg0_1,
            base,
            partial,
            ROWS=rows,
            COLS=cols,
            SEQ=seq,
            BLOCK_M=ROW_BLOCK,
            BLOCK_N=COL_BLOCK,
            num_warps=num_warps,
            num_stages=4,
        )
        _final_row_tile_sum_kernel[(triton.cdiv(cols, FINAL_BLOCK_C),)](
            partial,
            out_sum,
            NUM_ROW_TILES=num_row_tiles,
            COLS=cols,
            BLOCK_R=PARTIAL_BLOCK,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=num_warps,
            num_stages=4,
        )
    else:
        partial = torch.empty((batch, cols), device=arg0_1.device, dtype=torch.float32)
        _copy_partial_sum_kernel[(triton.cdiv(batch * cols, XBLOCK),)](
            arg0_1,
            base,
            partial,
            XNUMEL=batch * cols,
            COLS=cols,
            SEQ=seq,
            XBLOCK=XBLOCK,
            SBLOCK=SBLOCK,
            num_warps=num_warps,
            num_stages=4,
        )
        _final_sum_kernel[(triton.cdiv(cols, BLOCK_C),)](
            partial,
            out_sum,
            BATCH=batch,
            COLS=cols,
            BLOCK_B=BLOCK_B,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=4,
        )
    return base, base.permute(1, 0), out_sum
