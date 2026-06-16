"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT shifted-neighbor embedding assembly by gathering next-token, current-token, and previous-token bf16 embedding rows and writing the final contiguous `[32768, 384]` cat/view output directly with exact edge zero-fill, whereas Inductor lowers the embedding, two slices, two constant pads, cat, and view through generic gather/layout regions; Inductor cannot materially improve this local repro today because the full output contract is dominated by mandatory embedding-table reads and dense bf16 output stores rather than avoidable arithmetic or reductions; the fix is BANDWIDTH_BOUND: keep this as an at-floor gather/layout measurement unless broader embedding gather, allocation, or launch-overhead work moves both paths together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
    hidden_offsets = tl.arange(0, 128)
    row_mask = row_offsets < N_ROWS
    seq_offsets = row_offsets % SEQ_LEN

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
        mask=next_mask[:, None],
        other=0.0,
    )
    tl.store(
        out_ptr + row_offsets[:, None] * OUT_WIDTH + hidden_offsets[None, :],
        next_values,
        mask=row_mask[:, None],
    )

    current_values = tl.load(
        table_ptr + current_ids[:, None] * HIDDEN_SIZE + hidden_offsets[None, :],
        mask=current_mask[:, None],
        other=0.0,
    )
    tl.store(
        out_ptr + row_offsets[:, None] * OUT_WIDTH + HIDDEN_SIZE + hidden_offsets[None, :],
        current_values,
        mask=row_mask[:, None],
    )

    prev_values = tl.load(
        table_ptr + prev_ids[:, None] * HIDDEN_SIZE + hidden_offsets[None, :],
        mask=prev_mask[:, None],
        other=0.0,
    )
    tl.store(
        out_ptr + row_offsets[:, None] * OUT_WIDTH + 2 * HIDDEN_SIZE + hidden_offsets[None, :],
        prev_values,
        mask=row_mask[:, None],
    )


# 875862cf: (T([30522,128], bf16), T([256,128], i64), S([32768,384]))
@oracle_impl(hardware="B200", point="875862cf", BLOCK_M=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    table, ids, _shape_param_0 = inputs
    rows = 32768
    hidden = 128
    out_width = 384
    out = torch.empty_strided(
        (rows, out_width),
        (out_width, 1),
        device=table.device,
        dtype=torch.bfloat16,
    )
    _shifted_embedding_cat_kernel[(triton.cdiv(rows, BLOCK_M),)](
        table,
        ids,
        out,
        N_ROWS=rows,
        SEQ_LEN=128,
        HIDDEN_SIZE=hidden,
        OUT_WIDTH=out_width,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
