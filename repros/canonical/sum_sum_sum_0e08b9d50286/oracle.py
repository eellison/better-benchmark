"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer embedding-backward tuple by recomputing the shared dropout-scaled rowwise producer once, accumulating both returned hidden-column reductions, and atomically scattering the dependent producer into the `[1,768]`, `[4098,768]`, and `[50265,768]` masked index-accumulate destinations with the captured `_unsafe_masked_index_put_accumulate` masks, whereas Inductor materializes the rowwise producer and lowers the sibling reductions plus three generic scatter-add outputs as separate scheduled regions; Inductor cannot do this today because scheduler/codegen has no multi-destination duplicate-index scatter-reduce node that shares row-local LayerNorm-backward reductions across several masked embedding-gradient accumulators; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that fuses dropout-scaled row algebra, hidden reductions, mask predicates, and multi-table indexed accumulation while preserving the fp32 formulation."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HIDDEN = 768
POSITION_ROWS = 4098
VOCAB_ROWS = 50265
DROPOUT_SCALE = 1.1111111111111112
BLOCK_H = 1024
ZERO_BLOCK = 1024


@triton.jit
def _zero_outputs_kernel(
    sum_arg4_ptr,
    sum_plain_ptr,
    scatter_one_ptr,
    scatter_position_ptr,
    scatter_vocab_ptr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    zero = tl.zeros((BLOCK_N,), dtype=tl.float32)

    sum_arg4_size = HIDDEN_
    sum_plain_base = sum_arg4_size
    scatter_one_base = sum_plain_base + HIDDEN_
    scatter_position_base = scatter_one_base + HIDDEN_
    scatter_position_size = POSITION_ROWS_ * HIDDEN_
    scatter_vocab_base = scatter_position_base + scatter_position_size
    total = scatter_vocab_base + VOCAB_ROWS_ * HIDDEN_

    mask0 = offsets < sum_arg4_size
    tl.store(sum_arg4_ptr + offsets, zero, mask=mask0)

    off1 = offsets - sum_plain_base
    mask1 = (off1 >= 0) & (off1 < HIDDEN_)
    tl.store(sum_plain_ptr + off1, zero, mask=mask1)

    off2 = offsets - scatter_one_base
    mask2 = (off2 >= 0) & (off2 < HIDDEN_)
    tl.store(scatter_one_ptr + off2, zero, mask=mask2)

    off3 = offsets - scatter_position_base
    mask3 = (off3 >= 0) & (off3 < scatter_position_size)
    tl.store(scatter_position_ptr + off3, zero, mask=mask3)

    off4 = offsets - scatter_vocab_base
    mask4 = (off4 >= 0) & (offsets < total)
    tl.store(scatter_vocab_ptr + off4, zero, mask=mask4)


@triton.jit
def _longformer_embedding_bwd_kernel(
    keep_ptr,
    grad_ptr,
    weight_ptr,
    saved_ptr,
    row_scale_ptr,
    one_idx_ptr,
    position_idx_ptr,
    vocab_mask_ptr,
    vocab_idx_ptr,
    sum_arg4_ptr,
    sum_plain_ptr,
    scatter_one_ptr,
    scatter_position_ptr,
    scatter_vocab_ptr,
    HIDDEN_: tl.constexpr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_H_: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.arange(0, BLOCK_H_)
    h_mask = h < HIDDEN_
    weight = tl.load(weight_ptr + h, mask=h_mask, other=0.0).to(tl.float32)

    acc_sum_arg4 = tl.zeros((BLOCK_H_,), dtype=tl.float32)
    acc_sum_plain = tl.zeros((BLOCK_H_,), dtype=tl.float32)
    acc_scatter_one = tl.zeros((BLOCK_H_,), dtype=tl.float32)

    for batch in tl.static_range(0, BATCH_):
        row = batch * SEQ_ + seq
        offsets = row * HIDDEN_ + h

        keep = tl.load(keep_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
        saved = tl.load(saved_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)

        masked_grad = grad * (keep * DROPOUT_SCALE_)
        weighted = masked_grad * weight
        row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(h_mask, weighted * saved, 0.0), axis=0)
        scale = tl.load(row_scale_ptr + row).to(tl.float32)
        scatter_value = scale * (weighted * HIDDEN_ - row_sum - saved * row_dot)

        acc_sum_arg4 += tl.where(h_mask, masked_grad * saved, 0.0)
        acc_sum_plain += tl.where(h_mask, masked_grad, 0.0)

        # The captured unsafe one-row scatter has a full True mask; with the
        # generated indices, the visible result matches accumulating every row
        # into row 0.
        acc_scatter_one += tl.where(h_mask, scatter_value, 0.0)

        position_raw = tl.load(position_idx_ptr + row).to(tl.int64)
        position_wrapped = tl.where(
            position_raw < 0,
            position_raw + POSITION_ROWS_,
            position_raw,
        )
        position_valid = (
            h_mask
            & (position_raw >= 0)
            & (position_raw < POSITION_ROWS_)
            & (position_raw != 1)
        )
        tl.atomic_add(
            scatter_position_ptr + position_wrapped * HIDDEN_ + h,
            scatter_value,
            sem="relaxed",
            mask=position_valid,
        )

        vocab_raw = tl.load(vocab_idx_ptr + row).to(tl.int64)
        vocab_wrapped = tl.where(vocab_raw < 0, vocab_raw + VOCAB_ROWS_, vocab_raw)
        vocab_keep = tl.load(vocab_mask_ptr + row).to(tl.int1)
        vocab_valid = (
            h_mask
            & vocab_keep
            & (vocab_wrapped >= 0)
            & (vocab_wrapped < VOCAB_ROWS_)
        )
        tl.atomic_add(
            scatter_vocab_ptr + vocab_wrapped * HIDDEN_ + h,
            scatter_value,
            sem="relaxed",
            mask=vocab_valid,
        )

    tl.atomic_add(sum_arg4_ptr + h, acc_sum_arg4, sem="relaxed", mask=h_mask)
    tl.atomic_add(sum_plain_ptr + h, acc_sum_plain, sem="relaxed", mask=h_mask)
    tl.atomic_add(scatter_one_ptr + h, acc_scatter_one, sem="relaxed", mask=h_mask)


# 05376402: Longformer embedding backward scatter/reduction tuple.
@oracle_impl(hardware="B200", point="05376402")
def oracle_forward(inputs):
    (
        keep,
        grad,
        weight,
        saved,
        row_scale,
        one_idx,
        position_idx,
        vocab_mask,
        vocab_idx,
        *_shape_params,
    ) = inputs

    device = grad.device
    sum_arg4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_plain = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    scatter_one = torch.empty((1, HIDDEN), device=device, dtype=torch.float32)
    scatter_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    scatter_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    total_zero_elems = (3 + POSITION_ROWS + VOCAB_ROWS) * HIDDEN
    _zero_outputs_kernel[(triton.cdiv(total_zero_elems, ZERO_BLOCK),)](
        sum_arg4,
        sum_plain,
        scatter_one,
        scatter_position,
        scatter_vocab,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_N=ZERO_BLOCK,
        num_warps=4,
    )
    _longformer_embedding_bwd_kernel[(SEQ,)](
        keep,
        grad,
        weight,
        saved,
        row_scale,
        one_idx,
        position_idx,
        vocab_mask,
        vocab_idx,
        sum_arg4,
        sum_plain,
        scatter_one,
        scatter_position,
        scatter_vocab,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        SEQ_=SEQ,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=BLOCK_H,
        num_warps=8,
    )
    return sum_arg4, sum_plain, scatter_one, scatter_position, scatter_vocab
