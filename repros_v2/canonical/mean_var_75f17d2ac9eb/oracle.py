"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT bf16 word/position/token-type embedding assembly, returned bf16 summed embedding, hidden-size-768 unbiased variance normalization with the captured bf16 mean/sub/mul/sqrt/div boundaries, affine epilogue, and three aliasing `[2048, 768]` view returns in one Triton row kernel, whereas Inductor lowers the embedding gathers, visible summed-embedding producer, row reductions, bf16 normalization epilogue, and repeated alias-only views through generic schedules; Inductor cannot do this today because its scheduler does not carry indexed embedding gathers into a fixed-hidden normalization template while preserving visible bf16 rounding boundaries and aliasing view outputs; the fix is SCHEDULER_FUSION: teach normalization scheduling to fuse embedding gather producers and alias-view epilogues into one guarded row-normalization plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _embedding_layernorm_alias_kernel(
    word_table_ptr,
    input_ids_ptr,
    position_table_ptr,
    token_type_table_ptr,
    token_type_ids_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]
    seq_index = row_ids - (row_ids // SEQ) * SEQ

    word_ids = tl.load(input_ids_ptr + row_ids, mask=row_mask, other=0)
    token_type_ids = tl.load(token_type_ids_ptr + row_ids, mask=row_mask, other=0)
    word = tl.load(
        word_table_ptr + word_ids[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + seq_index[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    token_type = tl.load(
        token_type_table_ptr + token_type_ids[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    add0 = (word + position).to(tl.bfloat16).to(tl.float32)
    add1 = (add0 + token_type).to(tl.bfloat16).to(tl.float32)
    tl.store(add_out_ptr + offsets, add1.to(tl.bfloat16), mask=mask)

    add_for_reduce = tl.where(mask, add1, 0.0)
    sum_x = tl.sum(add_for_reduce, axis=1)[:, None]
    mean_f32 = sum_x / HIDDEN
    mean_bf16 = mean_f32.to(tl.bfloat16).to(tl.float32)

    centered = (add1 - mean_bf16).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    scaled = (weight[None, :] * centered).to(tl.bfloat16).to(tl.float32)

    sum_x2 = tl.sum(tl.where(mask, add1 * add1, 0.0), axis=1)[:, None]
    variance = (sum_x2 - sum_x * mean_f32) / (HIDDEN - 1.0)
    denom = (tl.sqrt(tl.maximum(variance, 0.0)).to(tl.bfloat16).to(tl.float32) + 1.0e-6)
    denom = denom.to(tl.bfloat16).to(tl.float32)

    divided = (scaled / denom).to(tl.bfloat16).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    out = (divided + bias[None, :]).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, out, mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a655df0f", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (
        word_table,
        input_ids,
        position_table,
        token_type_table,
        token_type_ids,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = inputs
    del shape1, shape2

    out_shape = _shape_tuple(shape0)
    batch = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq
    add_out = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_alias_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        word_table,
        input_ids,
        position_table,
        token_type_table,
        token_type_ids,
        weight,
        bias,
        add_out,
        norm_out,
        ROWS=rows,
        HIDDEN=hidden,
        SEQ=seq,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out, norm_out, norm_out
