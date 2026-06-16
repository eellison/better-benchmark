"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete ALBERT bf16 embedding-backward return tuple by fusing the shared rowwise hidden producer, both `[128]` sibling reductions, and all three duplicate-preserving masked scatter-add outputs into Triton kernels, whereas Inductor materializes the `[8,512,128]` producer and schedules the two reductions, zero fills, validity masks, 512-row position scatter, 2-row token-type scatter, 30000-row vocabulary scatter, and dense vocabulary add as separate generic regions; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that shares one row-local reduction producer across multiple indexed accumulation destinations while preserving bf16-to-fp32 casts, masked `_unsafe_masked_index_put_accumulate` semantics, duplicate-safe atomics, and the full dense output scope; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that keeps the row algebra in registers, emits the hidden-column reductions, folds validity masks into the indexed updates, and accumulates every embedding-gradient output directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 512
HIDDEN = 128
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 30000


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _init_outputs_kernel(
    base_vocab_ptr,
    position_out_ptr,
    segment_out_ptr,
    vocab_out_ptr,
    TOTAL_POSITION: tl.constexpr,
    TOTAL_SEGMENT: tl.constexpr,
    TOTAL_VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    vocab_mask = offsets < TOTAL_VOCAB
    base = tl.load(base_vocab_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(vocab_out_ptr + offsets, base, mask=vocab_mask)
    tl.store(position_out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < TOTAL_POSITION)
    tl.store(segment_out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < TOTAL_SEGMENT)


@triton.jit
def _seq_scatter_reduce_kernel(
    x_ptr,
    weight_ptr,
    normed_ptr,
    row_scale_ptr,
    position_idx_ptr,
    segment_idx_ptr,
    vocab_idx_ptr,
    partial_x_normed_ptr,
    partial_x_ptr,
    position_out_ptr,
    segment_out_ptr,
    vocab_out_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    BATCH_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
):
    seq = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_normed = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_position_grad = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for b in tl.static_range(0, BATCH_):
        row = b * SEQ_ + seq
        offsets = row * HIDDEN_ + cols
        mask = col_mask

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=True, other=0.0).to(tl.float32)

        weighted = _mul_rn(x, weight)
        scaled_weighted = _mul_rn(weighted, ROW_FACTOR_)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        weighted_normed = _mul_rn(weighted, normed)
        row_dot = tl.sum(tl.where(mask, weighted_normed, 0.0), axis=0)
        centered = _sub_rn(_sub_rn(scaled_weighted, row_sum), _mul_rn(normed, row_dot))
        grad = _mul_rn(row_scale, centered)

        acc_x_normed += tl.where(mask, _mul_rn(x, normed), 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_position_grad += tl.where(mask, grad, 0.0)

        segment_row = tl.load(segment_idx_ptr + seq).to(tl.int64)
        segment_active = (
            mask
            & (segment_row >= 0)
            & (segment_row < SEGMENT_ROWS_)
            & (segment_row != -1)
        )
        tl.atomic_add(
            segment_out_ptr + segment_row * HIDDEN_ + cols,
            grad,
            sem="relaxed",
            mask=segment_active,
        )

        vocab_row = tl.load(vocab_idx_ptr + row).to(tl.int64)
        vocab_active = (
            mask
            & (vocab_row >= 0)
            & (vocab_row < VOCAB_ROWS_)
            & (vocab_row != 0)
        )
        tl.atomic_add(
            vocab_out_ptr + vocab_row * HIDDEN_ + cols,
            grad,
            sem="relaxed",
            mask=vocab_active,
        )

    position_row = tl.load(position_idx_ptr + seq).to(tl.int64)
    position_active = (
        col_mask
        & (position_row >= 0)
        & (position_row < POSITION_ROWS_)
        & (position_row != -1)
    )
    tl.atomic_add(
        position_out_ptr + position_row * HIDDEN_ + cols,
        acc_position_grad,
        sem="relaxed",
        mask=position_active,
    )

    partial_offsets = seq * HIDDEN_ + cols
    tl.store(partial_x_normed_ptr + partial_offsets, acc_x_normed, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)


@triton.jit
def _finalize_hidden_sums_kernel(
    partial_x_normed_ptr,
    partial_x_ptr,
    out_x_normed_ptr,
    out_x_ptr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    seq = tl.arange(0, BLOCK_N)
    mask = (seq[:, None] < SEQ_) & (cols[None, :] < HIDDEN_)
    offsets = seq[:, None] * HIDDEN_ + cols[None, :]

    x_normed = tl.load(partial_x_normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out_mask = cols < HIDDEN_
    tl.store(out_x_normed_ptr + cols, tl.sum(x_normed, axis=0), mask=out_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=out_mask)


@oracle_impl(
    hardware="B200",
    point="409a14a3",
    HIDDEN_BLOCK=128,
    FINAL_BLOCK_H=16,
    INIT_BLOCK=1024,
    scatter_warps=4,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    HIDDEN_BLOCK: int,
    FINAL_BLOCK_H: int,
    INIT_BLOCK: int,
    scatter_warps: int,
    final_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        *_shape_params,
    ) = inputs

    device = arg1_1.device
    hidden = int(arg1_1.shape[1])
    position_out = torch.empty((POSITION_ROWS, hidden), device=device, dtype=torch.float32)
    segment_out = torch.empty((SEGMENT_ROWS, hidden), device=device, dtype=torch.float32)
    vocab_out = torch.empty((VOCAB_ROWS, hidden), device=device, dtype=torch.float32)
    partial_x_normed = torch.empty((SEQ, hidden), device=device, dtype=torch.float32)
    partial_x = torch.empty((SEQ, hidden), device=device, dtype=torch.float32)
    out_x_normed = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_x = torch.empty((hidden,), device=device, dtype=torch.float32)

    total_position = POSITION_ROWS * hidden
    total_segment = SEGMENT_ROWS * hidden
    total_vocab = VOCAB_ROWS * hidden
    _init_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        arg0_1,
        position_out,
        segment_out,
        vocab_out,
        TOTAL_POSITION=total_position,
        TOTAL_SEGMENT=total_segment,
        TOTAL_VOCAB=total_vocab,
        BLOCK=INIT_BLOCK,
        num_warps=4,
    )
    _seq_scatter_reduce_kernel[(SEQ,)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        partial_x_normed,
        partial_x,
        position_out,
        segment_out,
        vocab_out,
        HIDDEN_=hidden,
        SEQ_=SEQ,
        BATCH_=BATCH,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_H=HIDDEN_BLOCK,
        ROW_FACTOR_=float(HIDDEN),
        num_warps=scatter_warps,
    )
    _finalize_hidden_sums_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partial_x_normed,
        partial_x,
        out_x_normed,
        out_x,
        SEQ_=SEQ,
        HIDDEN_=hidden,
        BLOCK_N=triton.next_power_of_2(SEQ),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=final_warps,
    )

    return out_x_normed, out_x, position_out, segment_out, vocab_out
