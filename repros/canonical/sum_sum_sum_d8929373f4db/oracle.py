"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DistilBERT dropout/layernorm-backward embedding-gradient tuple by copying the bf16 vocabulary base directly into the returned f32 table, computing the shared rowwise producer with the captured bf16-to-f32 casts, accumulating the returned hidden-column reductions, batch-reducing the position-gradient producer before the `[512,768]` masked scatter, and atomically adding valid token-index contributors into the dense vocabulary output, whereas Inductor materializes the rowwise gradient producer and lowers the sibling reductions plus both `_unsafe_masked_index_put_accumulate` destinations and the dense base add as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured duplicate-index scatter-reduce node that shares rowwise layernorm reductions across masked position and vocabulary embedding-gradient accumulators with a dense base-update epilogue; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that fuses row producer arithmetic, validity masks, hidden reductions, batch-position accumulation, and direct dense base update while preserving bf16-to-f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
SEQ = 128
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 512
VOCAB_ROWS = 30522
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _init_outputs_kernel(
    base_ptr,
    out_sum3_ptr,
    out_sum4_ptr,
    out_position_ptr,
    out_vocab_ptr,
    VOCAB_TOTAL: tl.constexpr,
    POSITION_TOTAL: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)

    vocab_active = offsets < VOCAB_TOTAL
    base = tl.load(base_ptr + offsets, mask=vocab_active, other=0.0).to(tl.float32)
    tl.store(out_vocab_ptr + offsets, base, mask=vocab_active)

    position_active = offsets < POSITION_TOTAL
    hidden_active = offsets < HIDDEN_
    tl.store(out_position_ptr + offsets, zeros, mask=position_active)
    tl.store(out_sum3_ptr + offsets, zeros, mask=hidden_active)
    tl.store(out_sum4_ptr + offsets, zeros, mask=hidden_active)


@triton.jit
def _row_stats_kernel(
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    keep_ptr,
    weight_ptr,
    saved_ptr,
    position_saved_ptr,
    mean_ptr,
    invstd_ptr,
    row_sum_ptr,
    row_dot_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    active = h < HIDDEN_
    offsets = row * HIDDEN_ + h
    seq = row - (row // SEQ_) * SEQ_
    position_offsets = seq * HIDDEN_ + h

    source = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    keep = tl.load(keep_ptr + offsets, mask=active, other=0).to(tl.float32)
    dropout = keep * DROPOUT_SCALE_
    mul1 = source * dropout
    weight = tl.load(weight_ptr + h, mask=active, other=0.0).to(tl.float32)
    mul2 = mul1 * weight

    saved = tl.load(saved_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    position_saved = tl.load(position_saved_ptr + position_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row).to(tl.float32)
    invstd = tl.load(invstd_ptr + row).to(tl.float32)
    add3 = saved + position_saved
    sub = add3 - mean
    mul4 = sub * invstd

    row_sum = tl.sum(tl.where(active, mul2, 0.0), axis=0)
    row_dot = tl.sum(tl.where(active, mul2 * mul4, 0.0), axis=0)
    tl.store(row_sum_ptr + row, row_sum)
    tl.store(row_dot_ptr + row, row_dot)


@triton.jit
def _seq_scatter_reduce_kernel(
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    keep_ptr,
    weight_ptr,
    saved_ptr,
    position_saved_ptr,
    mean_ptr,
    invstd_ptr,
    position_index_ptr,
    token_index_ptr,
    row_sum_ptr,
    row_dot_ptr,
    out_sum3_ptr,
    out_sum4_ptr,
    out_position_ptr,
    out_vocab_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    BATCH_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    seq = tl.program_id(1)
    pid_b = tl.program_id(2)

    batch_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = batch_offsets * SEQ_ + seq
    active = (batch_offsets[:, None] < BATCH_) & (cols[None, :] < HIDDEN_)
    row_active = batch_offsets < BATCH_
    channel_active = cols < HIDDEN_

    offsets = rows[:, None] * HIDDEN_ + cols[None, :]
    position_saved_offsets = seq * HIDDEN_ + cols

    source = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    keep = tl.load(keep_ptr + offsets, mask=active, other=0).to(tl.float32)
    dropout = keep * DROPOUT_SCALE_
    mul1 = source * dropout
    weight = tl.load(weight_ptr + cols, mask=channel_active, other=0.0).to(tl.float32)
    mul2 = mul1 * weight[None, :]

    saved = tl.load(saved_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    position_saved = tl.load(
        position_saved_ptr + position_saved_offsets,
        mask=channel_active,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    add3 = saved + position_saved[None, :]
    sub = add3 - mean[:, None]
    mul4 = sub * invstd[:, None]

    row_sum = tl.load(row_sum_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    mul3 = mul2 * HIDDEN_
    sub1 = mul3 - row_sum[:, None]
    mul6 = mul4 * row_dot[:, None]
    sub2 = sub1 - mul6
    div = invstd[:, None] / HIDDEN_
    producer = tl.where(active, div * sub2, 0.0)

    tl.atomic_add(
        out_sum3_ptr + cols,
        tl.sum(tl.where(active, mul1 * mul4, 0.0), axis=0),
        sem="relaxed",
        mask=channel_active,
    )
    tl.atomic_add(
        out_sum4_ptr + cols,
        tl.sum(tl.where(active, mul1, 0.0), axis=0),
        sem="relaxed",
        mask=channel_active,
    )

    position_index = tl.load(position_index_ptr + seq).to(tl.int64)
    position_valid = (
        channel_active
        & (position_index >= 0)
        & (position_index < POSITION_ROWS_)
        & (position_index != -1)
    )
    tl.atomic_add(
        out_position_ptr + position_index * HIDDEN_ + cols,
        tl.sum(producer, axis=0),
        sem="relaxed",
        mask=position_valid,
    )

    token_index = tl.load(token_index_ptr + rows, mask=row_active, other=-1).to(tl.int64)
    token_valid = (
        active
        & (token_index[:, None] >= 0)
        & (token_index[:, None] < VOCAB_ROWS_)
        & (token_index[:, None] != 0)
    )
    tl.atomic_add(
        out_vocab_ptr + token_index[:, None] * HIDDEN_ + cols[None, :],
        producer,
        sem="relaxed",
        mask=token_valid,
    )


@oracle_impl(
    hardware="B200",
    point="b2f45dd0",
    INIT_BLOCK=1024,
    BLOCK_H=1024,
    BLOCK_B=16,
    BLOCK_C=32,
    num_warps_init=4,
    num_warps_stats=8,
    num_warps_scatter=4,
)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    BLOCK_B: int,
    BLOCK_C: int,
    num_warps_init: int,
    num_warps_stats: int,
    num_warps_scatter: int,
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

    device = arg0_1.device
    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    out_sum3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    vocab_total = VOCAB_ROWS * HIDDEN
    position_total = POSITION_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(vocab_total, INIT_BLOCK),)](
        arg0_1,
        out_sum3,
        out_sum4,
        out_position,
        out_vocab,
        VOCAB_TOTAL=vocab_total,
        POSITION_TOTAL=position_total,
        HIDDEN_=HIDDEN,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _row_stats_kernel[(ROWS,)](
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
        row_sum,
        row_dot,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_stats,
    )

    _seq_scatter_reduce_kernel[
        (triton.cdiv(HIDDEN, BLOCK_C), SEQ, triton.cdiv(BATCH, BLOCK_B))
    ](
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
        row_sum,
        row_dot,
        out_sum3,
        out_sum4,
        out_position,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        VOCAB_ROWS_=VOCAB_ROWS,
        POSITION_ROWS_=POSITION_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_B=BLOCK_B,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_scatter,
    )

    return out_sum3, out_sum4, out_position, out_vocab
