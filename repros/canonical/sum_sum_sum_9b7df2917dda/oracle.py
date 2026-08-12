"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Roberta dropout/layernorm-backward embedding-gradient tuple by copying the bf16 vocabulary base slice directly to the returned f32 table, computing the shared rowwise producer once, accumulating both returned hidden-column reductions, and atomically adding that producer into the position, segment, and token embedding-gradient outputs with the captured validity masks, whereas Inductor materializes the rowwise pointwise/reduction producer and lowers the three `_unsafe_masked_index_put_accumulate` destinations plus sibling reductions as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured duplicate-index scatter-reduce node that shares rowwise hidden reductions across multiple masked embedding-gradient accumulators and a dense base-update epilogue; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that fuses row producer arithmetic, validity masks, hidden reductions, and direct dense base update while preserving bf16-to-f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 50265
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _init_outputs_kernel(
    base_ptr,
    out_sum3_ptr,
    out_sum4_ptr,
    out_position_ptr,
    out_segment_ptr,
    out_vocab_ptr,
    VOCAB_TOTAL: tl.constexpr,
    POSITION_TOTAL: tl.constexpr,
    SEGMENT_TOTAL: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)

    vocab_mask = offsets < VOCAB_TOTAL
    base = tl.load(base_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(out_vocab_ptr + offsets, base, mask=vocab_mask)

    position_mask = offsets < POSITION_TOTAL
    segment_mask = offsets < SEGMENT_TOTAL
    hidden_mask = offsets < HIDDEN_
    tl.store(out_position_ptr + offsets, zeros, mask=position_mask)
    tl.store(out_segment_ptr + offsets, zeros, mask=segment_mask)
    tl.store(out_sum3_ptr + offsets, zeros, mask=hidden_mask)
    tl.store(out_sum4_ptr + offsets, zeros, mask=hidden_mask)


@triton.jit
def _row_scatter_reduce_kernel(
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    keep_ptr,
    weight_ptr,
    saved_ptr,
    scale_ptr,
    position_index_ptr,
    segment_index_ptr,
    vocab_mask_ptr,
    vocab_index_ptr,
    out_sum3_ptr,
    out_sum4_ptr,
    out_position_ptr,
    out_segment_ptr,
    out_vocab_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    active = (row < ROWS_) & (h < HIDDEN_)
    offsets = row * HIDDEN_ + h

    added = (
        tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    )
    keep = tl.load(keep_ptr + offsets, mask=active, other=0).to(tl.float32)
    mul1 = added * (keep * DROPOUT_SCALE_)

    weight = tl.load(weight_ptr + h, mask=h < HIDDEN_, other=0.0).to(tl.float32)
    saved = tl.load(saved_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mul2 = mul1 * weight
    row_sum = tl.sum(tl.where(active, mul2, 0.0), axis=0)
    row_dot = tl.sum(tl.where(active, mul2 * saved, 0.0), axis=0)

    scale = tl.load(scale_ptr + row, mask=row < ROWS_, other=0.0).to(tl.float32)
    producer = scale * ((mul2 * HIDDEN_) - row_sum - (saved * row_dot))

    tl.atomic_add(out_sum3_ptr + h, mul1 * saved, sem="relaxed", mask=active)
    tl.atomic_add(out_sum4_ptr + h, mul1, sem="relaxed", mask=active)

    position_index = tl.load(position_index_ptr + row, mask=row < ROWS_, other=-1).to(tl.int64)
    position_active = (
        (position_index >= 0)
        & (position_index < POSITION_ROWS_)
        & (position_index != 0)
    )
    tl.atomic_add(
        out_position_ptr + position_index * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & position_active,
    )

    segment_index = tl.load(segment_index_ptr + row, mask=row < ROWS_, other=-1).to(tl.int64)
    segment_active = (
        (segment_index >= 0)
        & (segment_index < SEGMENT_ROWS_)
        & (segment_index != -1)
    )
    tl.atomic_add(
        out_segment_ptr + segment_index * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & segment_active,
    )

    vocab_mask = tl.load(vocab_mask_ptr + row, mask=row < ROWS_, other=0)
    vocab_index = tl.load(vocab_index_ptr + row, mask=row < ROWS_, other=-1).to(tl.int64)
    vocab_active = vocab_mask & (vocab_index >= 0) & (vocab_index < VOCAB_ROWS_)
    tl.atomic_add(
        out_vocab_ptr + vocab_index * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & vocab_active,
    )


@oracle_impl(hardware="B200", point="14a5f5ad", INIT_BLOCK=1024, BLOCK_H=1024, num_warps_init=4, num_warps_row=8)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    num_warps_init: int,
    num_warps_row: int,
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
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        *_shape_params,
    ) = inputs

    out_sum3 = torch.empty((HIDDEN,), device=arg0_1.device, dtype=torch.float32)
    out_sum4 = torch.empty((HIDDEN,), device=arg0_1.device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=arg0_1.device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, HIDDEN), device=arg0_1.device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=arg0_1.device, dtype=torch.float32)

    vocab_total = VOCAB_ROWS * HIDDEN
    position_total = POSITION_ROWS * HIDDEN
    segment_total = SEGMENT_ROWS * HIDDEN
    init_total = vocab_total
    _init_outputs_kernel[(triton.cdiv(init_total, INIT_BLOCK),)](
        arg0_1,
        out_sum3,
        out_sum4,
        out_position,
        out_segment,
        out_vocab,
        VOCAB_TOTAL=vocab_total,
        POSITION_TOTAL=position_total,
        SEGMENT_TOTAL=segment_total,
        HIDDEN_=HIDDEN,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _row_scatter_reduce_kernel[(ROWS,)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        out_sum3,
        out_sum4,
        out_position,
        out_segment,
        out_vocab,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_row,
    )

    return out_sum3, out_sum4, out_position, out_segment, out_vocab
