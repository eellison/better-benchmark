"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete TrOCR generated-position embedding scope in one Triton broadcast-copy kernel, including the `prims.iota(256)`, `[64,256]` expand, constant `+2` int64 index tensor returned by the repro, and fresh contiguous fp32 `[64,256,1024]` embedding output; Inductor lowers the generated index tensor and `aten.embedding.default` through generic pointwise/indexing code even though the row range is affine and static; the fix is NEW_PATTERN: add a guarded generated-position embedding lowering that proves the constant row offset, emits the returned indices, and broadcasts copied table rows directly into the final layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ_LEN = 256
ROW_OFFSET = 2


@triton.jit
def _generated_position_embedding_kernel(
    table_ptr,
    index_out_ptr,
    embedding_out_ptr,
    table_stride_0: tl.constexpr,
    table_stride_1: tl.constexpr,
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    hidden: tl.constexpr,
    row_offset: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    token = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_block = tl.program_id(2)

    cols = col_block * BLOCK_D + tl.arange(0, BLOCK_D)
    batches = batch_block * BLOCK_B + tl.arange(0, BLOCK_B)
    col_mask = cols < hidden
    batch_mask = batches < batch

    row = token + row_offset
    values = tl.load(
        table_ptr + row * table_stride_0 + cols * table_stride_1,
        mask=col_mask,
        other=0.0,
    )

    embedding_offsets = (batches[:, None] * seq_len + token) * hidden + cols[None, :]
    tl.store(
        embedding_out_ptr + embedding_offsets,
        values[None, :],
        mask=batch_mask[:, None] & col_mask[None, :],
    )

    if col_block == 0:
        index_values = tl.zeros((BLOCK_B,), tl.int64) + row
        tl.store(
            index_out_ptr + batches * seq_len + token,
            index_values,
            mask=batch_mask,
        )


@oracle_impl(hardware="B200", point="59aa3e3a", BLOCK_D=1024, BLOCK_B=4, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_D: int, BLOCK_B: int, num_warps: int, num_stages: int):
    table, shape_param = inputs
    batch = int(shape_param[0])
    hidden = int(table.shape[1])

    indices = torch.empty_strided(
        (batch, SEQ_LEN),
        (SEQ_LEN, 1),
        device=table.device,
        dtype=torch.int64,
    )
    embedding = torch.empty_strided(
        (batch, SEQ_LEN, hidden),
        (SEQ_LEN * hidden, hidden, 1),
        device=table.device,
        dtype=torch.float32,
    )

    grid = (SEQ_LEN, triton.cdiv(hidden, BLOCK_D), triton.cdiv(batch, BLOCK_B))
    _generated_position_embedding_kernel[grid](
        table,
        indices,
        embedding,
        table_stride_0=table.stride(0),
        table_stride_1=table.stride(1),
        batch=batch,
        seq_len=SEQ_LEN,
        hidden=hidden,
        row_offset=ROW_OFFSET,
        BLOCK_D=BLOCK_D,
        BLOCK_B=BLOCK_B,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return indices, embedding
