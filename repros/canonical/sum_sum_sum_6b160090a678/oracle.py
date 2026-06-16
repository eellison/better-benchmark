"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Electra bf16 dropout/layer-norm-backward embedding-gradient tuple by reconstructing the fixed hidden-size row producer once, feeding both returned hidden reductions, the batch-reduced position scatter, the token-type scatter, and the vocabulary scatter added to the bf16 base table directly from that producer, whereas Inductor materializes the rowwise producer and schedules the reductions, validity masks, three `_unsafe_masked_index_put_accumulate` destinations, and final dense add as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that shares a row-reduction epilogue across multiple duplicate-index accumulate users while preserving masked value semantics and bf16-to-f32 cast boundaries; the fix is SCATTER_REDUCE: add a guarded embedding-backward scatter-reduce lowering that initializes dense destinations directly, folds validity masks into indexed updates, and finalizes compatible hidden reductions from the same row producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 64
SEQ = 512
HIDDEN = 128
ROWS = BATCH * SEQ
POSITION_ROWS = 512
TYPE_ROWS = 2
VOCAB_ROWS = 30522
DROP_SCALE = 1.1111111111111112


@triton.jit
def _init_outputs_kernel(
    base_ptr,
    sum3_ptr,
    sum4_ptr,
    position_ptr,
    type_ptr,
    vocab_ptr,
    VOCAB_N: tl.constexpr,
    POSITION_N: tl.constexpr,
    TYPE_N: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)

    vocab_mask = offsets < VOCAB_N
    base = tl.load(base_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(vocab_ptr + offsets, base, mask=vocab_mask)
    tl.store(position_ptr + offsets, zeros, mask=offsets < POSITION_N)
    tl.store(type_ptr + offsets, zeros, mask=offsets < TYPE_N)
    tl.store(sum3_ptr + offsets, zeros, mask=offsets < HIDDEN_)
    tl.store(sum4_ptr + offsets, zeros, mask=offsets < HIDDEN_)


@triton.jit
def _seq_reduce_scatter_kernel(
    x_ptr,
    keep_ptr,
    weight_ptr,
    normed_ptr,
    row_scale_ptr,
    position_index_ptr,
    type_index_ptr,
    sum3_ptr,
    sum4_ptr,
    position_ptr,
    type_ptr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    TYPE_ROWS_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seq = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    pos_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    type_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    sum3_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    sum4_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for batch in tl.static_range(0, BATCH_):
        row = batch * SEQ_ + seq
        offsets = row * HIDDEN_ + cols
        x = tl.load(x_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=col_mask, other=0).to(tl.float32)
        normed = tl.load(normed_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row).to(tl.float32)

        mul1 = x * (keep * DROP_SCALE_)
        weighted = mul1 * weight
        row_sum = tl.sum(tl.where(col_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(col_mask, weighted * normed, 0.0), axis=0)
        producer = row_scale * (weighted * HIDDEN_ - row_sum - normed * row_dot)

        pos_acc += tl.where(col_mask, producer, 0.0)
        type_acc += tl.where(col_mask, producer, 0.0)
        sum3_acc += tl.where(col_mask, mul1 * normed, 0.0)
        sum4_acc += tl.where(col_mask, mul1, 0.0)

    position_raw = tl.load(position_index_ptr + seq).to(tl.int64)
    position_valid = (
        col_mask
        & (position_raw >= 0)
        & (position_raw < POSITION_ROWS_)
        & (position_raw != -1)
    )
    tl.atomic_add(
        position_ptr + position_raw * HIDDEN_ + cols,
        pos_acc,
        sem="relaxed",
        mask=position_valid,
    )

    type_raw = tl.load(type_index_ptr + seq).to(tl.int64)
    type_valid = col_mask & (type_raw >= 0) & (type_raw < TYPE_ROWS_) & (type_raw != -1)
    tl.atomic_add(
        type_ptr + type_raw * HIDDEN_ + cols,
        type_acc,
        sem="relaxed",
        mask=type_valid,
    )

    tl.atomic_add(sum3_ptr + cols, sum3_acc, sem="relaxed", mask=col_mask)
    tl.atomic_add(sum4_ptr + cols, sum4_acc, sem="relaxed", mask=col_mask)


@triton.jit
def _vocab_scatter_kernel(
    x_ptr,
    keep_ptr,
    weight_ptr,
    normed_ptr,
    row_scale_ptr,
    vocab_index_ptr,
    vocab_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = rows < ROWS_
    col_mask = cols < HIDDEN_
    mask = row_mask & col_mask
    offsets = rows * HIDDEN_ + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    mul1 = x * (keep * DROP_SCALE_)
    weighted = mul1 * weight
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)[:, None]
    row_dot = tl.sum(tl.where(mask, weighted * normed, 0.0), axis=1)[:, None]
    producer = row_scale * (weighted * HIDDEN_ - row_sum - normed * row_dot)

    vocab_raw = tl.load(vocab_index_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    vocab_valid = (
        mask
        & (vocab_raw >= 0)
        & (vocab_raw < VOCAB_ROWS_)
        & (vocab_raw != 0)
    )
    tl.atomic_add(
        vocab_ptr + vocab_raw * HIDDEN_ + cols,
        producer,
        sem="relaxed",
        mask=vocab_valid,
    )


@oracle_impl(
    hardware="B200",
    point="094119a0",
    INIT_BLOCK=1024,
    BLOCK_H=128,
    BLOCK_M=1,
    num_warps_init=4,
    num_warps_seq=4,
    num_warps_vocab=4,
)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    BLOCK_M: int,
    num_warps_init: int,
    num_warps_seq: int,
    num_warps_vocab: int,
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
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    out_sum3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_type = torch.empty((TYPE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    vocab_n = VOCAB_ROWS * HIDDEN
    position_n = POSITION_ROWS * HIDDEN
    type_n = TYPE_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(vocab_n, INIT_BLOCK),)](
        arg0_1,
        out_sum3,
        out_sum4,
        out_position,
        out_type,
        out_vocab,
        VOCAB_N=vocab_n,
        POSITION_N=position_n,
        TYPE_N=type_n,
        HIDDEN_=HIDDEN,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _seq_reduce_scatter_kernel[(SEQ,)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        out_sum3,
        out_sum4,
        out_position,
        out_type,
        BATCH_=BATCH,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        TYPE_ROWS_=TYPE_ROWS,
        DROP_SCALE_=DROP_SCALE,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_seq,
    )

    _vocab_scatter_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg8_1,
        out_vocab,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROP_SCALE_=DROP_SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_vocab,
    )

    return out_sum3, out_sum4, out_position, out_type, out_vocab
