"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GPT-Neo bf16 layernorm-backward embedding-gradient tuple by zero-initializing both returned scatter tables, recomputing the shared rowwise producer with the captured bf16-to-f32 casts, accumulating both returned hidden-column reductions, batch-reducing the position-index producer before the `[2048,2048]` masked scatter, and atomically adding valid token-index contributors into the `[50257,2048]` vocabulary output, whereas Inductor materializes the rowwise producer and lowers the sibling reductions plus both `_unsafe_masked_index_put_accumulate` destinations as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured duplicate-index scatter-reduce node that shares row-local layernorm reductions across a batch-reduced position accumulator and a token-index vocabulary accumulator; the fix is SCATTER_REDUCE: add a GPT-Neo embedding-backward scatter-reduce lowering that fuses row algebra, validity masks, hidden reductions, batch-position accumulation, and direct token scatter while preserving bf16-to-f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 2048
POSITION_ROWS = 2048
VOCAB_ROWS = 50257


@triton.jit
def _init_outputs_kernel(
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
    tl.store(out_vocab_ptr + offsets, zeros, mask=vocab_active)

    position_active = offsets < POSITION_TOTAL
    hidden_active = offsets < HIDDEN_
    tl.store(out_position_ptr + offsets, zeros, mask=position_active)
    tl.store(out_sum3_ptr + offsets, zeros, mask=hidden_active)
    tl.store(out_sum4_ptr + offsets, zeros, mask=hidden_active)


@triton.jit
def _row_stats_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    weight_ptr,
    saved_a_ptr,
    saved_b_ptr,
    mean_ptr,
    invstd_ptr,
    row_sum_ptr,
    row_dot_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    active = h < HIDDEN_
    seq = row - (row // SEQ_) * SEQ_
    offsets = row * HIDDEN_ + h
    broadcast_offsets = seq * HIDDEN_ + h

    source = tl.load(arg0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    weight = tl.load(weight_ptr + h, mask=active, other=0.0).to(tl.float32)
    weighted = source * weight

    norm = tl.load(saved_a_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    norm += tl.load(saved_b_ptr + broadcast_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row).to(tl.float32)
    norm -= mean
    invstd = tl.load(invstd_ptr + row).to(tl.float32)
    norm *= invstd

    row_sum = tl.sum(tl.where(active, weighted, 0.0), axis=0)
    row_dot = tl.sum(tl.where(active, weighted * norm, 0.0), axis=0)
    tl.store(row_sum_ptr + row, row_sum)
    tl.store(row_dot_ptr + row, row_dot)


@triton.jit
def _seq_scatter_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    weight_ptr,
    saved_a_ptr,
    saved_b_ptr,
    mean_ptr,
    invstd_ptr,
    upstream_ptr,
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
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    seq = tl.program_id(1)

    batches = tl.arange(0, BLOCK_B)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = batches * SEQ_ + seq
    active = (batches[:, None] < BATCH_) & (cols[None, :] < HIDDEN_)
    row_active = batches < BATCH_
    col_active = cols < HIDDEN_

    offsets = rows[:, None] * HIDDEN_ + cols[None, :]
    broadcast_offsets = seq * HIDDEN_ + cols

    source = tl.load(arg0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    weight = tl.load(weight_ptr + cols, mask=col_active, other=0.0).to(tl.float32)
    weighted = source * weight[None, :]

    norm = tl.load(saved_a_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    norm += tl.load(saved_b_ptr + broadcast_offsets, mask=col_active, other=0.0).to(tl.float32)[None, :]
    mean = tl.load(mean_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    norm -= mean[:, None]
    invstd = tl.load(invstd_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    norm *= invstd[:, None]

    row_sum = tl.load(row_sum_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    mul_scaled = weighted * HIDDEN_
    sub1 = mul_scaled - row_sum[:, None]
    mul4 = norm * row_dot[:, None]
    sub2 = sub1 - mul4
    div = invstd[:, None] / HIDDEN_
    producer = div * sub2
    add3 = tl.load(upstream_ptr + offsets, mask=active, other=0.0).to(tl.float32) + producer

    tl.atomic_add(
        out_sum3_ptr + cols,
        tl.sum(tl.where(active, source * norm, 0.0), axis=0),
        sem="relaxed",
        mask=col_active,
    )
    tl.atomic_add(
        out_sum4_ptr + cols,
        tl.sum(tl.where(active, source, 0.0), axis=0),
        sem="relaxed",
        mask=col_active,
    )

    position_index = tl.load(position_index_ptr + seq).to(tl.int64)
    tl.atomic_add(
        out_position_ptr + position_index * HIDDEN_ + cols,
        tl.sum(tl.where(active, add3, 0.0), axis=0),
        sem="relaxed",
        mask=col_active,
    )

    token_index = tl.load(token_index_ptr + rows, mask=row_active, other=-1).to(tl.int64)
    token_valid = (
        (token_index[:, None] >= 0)
        & (token_index[:, None] < VOCAB_ROWS_)
        & (token_index[:, None] != -1)
    )
    tl.atomic_add(
        out_vocab_ptr + token_index[:, None] * HIDDEN_ + cols[None, :],
        add3,
        sem="relaxed",
        mask=active & token_valid,
    )


@oracle_impl(
    hardware="B200",
    point="cb38dfd8",
    INIT_BLOCK=1024,
    BLOCK_H=2048,
    BLOCK_B=32,
    BLOCK_C=32,
    num_warps_init=4,
    num_warps_stats=8,
    num_warps_scatter=8,
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
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        row_sum,
        row_dot,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_stats,
    )

    _seq_scatter_reduce_kernel[(triton.cdiv(HIDDEN, BLOCK_C), SEQ)](
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
        BLOCK_B=BLOCK_B,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_scatter,
    )

    return out_sum3, out_sum4, out_position, out_vocab
