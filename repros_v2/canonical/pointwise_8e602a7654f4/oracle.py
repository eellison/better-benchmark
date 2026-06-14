"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT shifted-neighbor embedding assembly by gathering next-token, current-token, and previous-token rows from the f32 embedding table and writing the final contiguous bf16[32768,384] cat/view output directly with border zero-fill, whereas Inductor lowers the captured embedding/slice/constant_pad_nd/cat/cast/view graph through generic gather and pointwise layout materialization; Inductor cannot do this today because it has no canonical shifted-token embedding-cat lowering that sinks the padding, concatenation, and bf16 output cast into the indexed gather schedule; the fix is NEW_PATTERN: add a guarded shifted-neighbor embedding lowering that emits the direct gathered output layout while preserving the border zeros, f32 table reads, bf16 cast, and returned output stride."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
SEQ = 128
HIDDEN = 128
ROWS = BATCH * SEQ
OUT_HIDDEN = HIDDEN * 3


@triton.jit
def _shifted_embedding_cat_bf16_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    seq_len: tl.constexpr,
    hidden: tl.constexpr,
    out_hidden: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_offsets = tl.arange(0, hidden)
    row_mask = row_offsets < n_rows
    seq_offsets = row_offsets - (row_offsets // seq_len) * seq_len

    next_mask = row_mask & (seq_offsets < (seq_len - 1))
    current_mask = row_mask
    prev_mask = row_mask & (seq_offsets > 0)

    next_positions = tl.where(next_mask, row_offsets + 1, row_offsets)
    current_positions = row_offsets
    prev_positions = tl.where(prev_mask, row_offsets - 1, row_offsets)

    next_ids = tl.load(ids_ptr + next_positions, mask=next_mask, other=0)
    current_ids = tl.load(ids_ptr + current_positions, mask=current_mask, other=0)
    prev_ids = tl.load(ids_ptr + prev_positions, mask=prev_mask, other=0)

    next_values = tl.load(
        table_ptr + next_ids[:, None] * hidden + hidden_offsets[None, :],
        mask=next_mask[:, None],
        other=0.0,
    )
    tl.store(
        out_ptr + row_offsets[:, None] * out_hidden + hidden_offsets[None, :],
        next_values.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=row_mask[:, None],
    )

    current_values = tl.load(
        table_ptr + current_ids[:, None] * hidden + hidden_offsets[None, :],
        mask=current_mask[:, None],
        other=0.0,
    )
    tl.store(
        out_ptr + row_offsets[:, None] * out_hidden + hidden + hidden_offsets[None, :],
        current_values.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=row_mask[:, None],
    )

    prev_values = tl.load(
        table_ptr + prev_ids[:, None] * hidden + hidden_offsets[None, :],
        mask=prev_mask[:, None],
        other=0.0,
    )
    tl.store(
        out_ptr + row_offsets[:, None] * out_hidden + 2 * hidden + hidden_offsets[None, :],
        prev_values.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=row_mask[:, None],
    )


@oracle_impl(hardware="B200", point="b3d88053", BLOCK_M=4, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    table, ids, shape = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape),
        (OUT_HIDDEN, 1),
        device=table.device,
        dtype=torch.bfloat16,
    )

    _shifted_embedding_cat_bf16_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        table,
        ids,
        out,
        n_rows=ROWS,
        seq_len=SEQ,
        hidden=HIDDEN,
        out_hidden=OUT_HIDDEN,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
