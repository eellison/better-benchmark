"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT bf16 QKV crop/cat/view/permute/clone layout, returns both the contiguous `[rows, cols]` clone view and its `[cols, rows]` transpose view, and accumulates the sibling f32 column sum with the required bf16 round-trip from the same source tiles, whereas Inductor materializes the cropped QKV clone and then schedules the column reduction as separate work over that materialized buffer; Inductor cannot do this today because its scheduler does not fuse a mandatory layout-changing clone producer with a compatible sibling reduction consumer while preserving multi-output view aliases and dtype cast boundaries; the fix is SCHEDULER_FUSION: teach layout materialization scheduling to emit same-tile partial reductions for sibling reductions of the cloned buffer and return the required aliasing views directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _materialize_partial_sum_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    matrix_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    IN_WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_in_chunk = tl.arange(0, RBLOCK)
    rows = tl.program_id(1) * CHUNK_ROWS + row_in_chunk[:, None]
    cols = col_vec[None, :]

    source_cols = HEADS * OUT_WIDTH
    source = cols // source_cols
    source_inner = cols - source * source_cols
    head = source_inner // OUT_WIDTH
    width = source_inner - head * OUT_WIDTH

    batch = rows // TOKENS
    token = rows - batch * TOKENS
    input_offsets = ((batch * HEADS + head) * TOKENS + token) * IN_WIDTH + width

    mask = (row_in_chunk[:, None] < CHUNK_ROWS) & (rows < ROWS) & (cols < COLS)
    vals2 = tl.load(x2_ptr + input_offsets, mask=mask & (source == 0), other=0.0)
    vals1 = tl.load(x1_ptr + input_offsets, mask=mask & (source == 1), other=0.0)
    vals0 = tl.load(x0_ptr + input_offsets, mask=mask & (source == 2), other=0.0)
    values = tl.where(source == 0, vals2, tl.where(source == 1, vals1, vals0))

    tl.store(matrix_ptr + rows * COLS + cols, values, mask=mask)
    partial = tl.sum(tl.where(mask, values.to(tl.float32), 0.0), axis=0)
    tl.store(
        partial_ptr + tl.program_id(1) * COLS + col_vec,
        partial,
        mask=col_vec < COLS,
    )


@triton.jit
def _finish_sum_kernel(
    partial_ptr,
    out_ptr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    col_vec = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    chunks = tl.arange(0, RBLOCK)
    mask = (chunks[:, None] < CHUNKS) & (col_vec[None, :] < COLS)
    values = tl.load(
        partial_ptr + chunks[:, None] * COLS + col_vec[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    totals = tl.sum(values, axis=0)
    rounded = totals.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + col_vec, rounded, mask=col_vec < COLS)


def _run(
    inputs,
    *,
    rows: int,
    cols: int,
    tokens: int,
    in_width: int,
    out_width: int,
    chunks: int,
    chunk_rows: int,
    block_cols: int,
    rblock: int,
    final_block_cols: int,
    final_rblock: int,
    num_warps: int,
):
    x0, x1, x2, *_shape_params = inputs
    matrix = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty((chunks, cols), device=x0.device, dtype=torch.float32)
    sums = torch.empty((cols,), device=x0.device, dtype=torch.float32)

    grid = (triton.cdiv(cols, block_cols), chunks)
    _materialize_partial_sum_kernel[grid](
        x0,
        x1,
        x2,
        matrix,
        partial,
        ROWS=rows,
        COLS=cols,
        TOKENS=tokens,
        HEADS=4,
        IN_WIDTH=in_width,
        OUT_WIDTH=out_width,
        CHUNK_ROWS=chunk_rows,
        BLOCK_COLS=block_cols,
        RBLOCK=rblock,
        num_warps=num_warps,
        num_stages=3,
    )
    _finish_sum_kernel[(triton.cdiv(cols, final_block_cols),)](
        partial,
        sums,
        COLS=cols,
        CHUNKS=chunks,
        BLOCK_COLS=final_block_cols,
        RBLOCK=final_rblock,
        num_warps=4,
        num_stages=3,
    )
    return matrix, matrix.permute(1, 0), sums


# 89a27d2d: (T([512,4,256,40], bf16), T([512,4,256,40], bf16), T([512,4,256,40], bf16), ...)
@oracle_impl(
    hardware="B200",
    point="89a27d2d",
    rows=131072,
    cols=432,
    tokens=256,
    in_width=40,
    out_width=36,
    chunks=256,
    chunk_rows=512,
    block_cols=32,
    rblock=512,
    final_block_cols=32,
    final_rblock=256,
    num_warps=8,
)
# 38bf62e8: (T([512,4,16,64], bf16), T([512,4,16,64], bf16), T([512,4,16,64], bf16), ...)
@oracle_impl(
    hardware="B200",
    point="38bf62e8",
    rows=8192,
    cols=720,
    tokens=16,
    in_width=64,
    out_width=60,
    chunks=64,
    chunk_rows=128,
    block_cols=32,
    rblock=128,
    final_block_cols=32,
    final_rblock=64,
    num_warps=4,
)
def oracle_forward(inputs, **kwargs):
    return _run(inputs, **kwargs)
