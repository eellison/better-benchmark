"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DistillGPT2 bf16 layernorm-backward/dropout embedding-gradient tuple by copying the bf16 vocabulary base directly into the returned f32 table, reconstructing the shared rowwise producer once, accumulating both returned hidden-column reductions, batch-reducing the position-gradient producer before the `[1024,768]` masked scatter, and atomically adding valid token-index contributors into the `[50257,768]` vocabulary output, whereas Inductor materializes the rowwise producer and lowers the sibling reductions plus both duplicate-preserving `_unsafe_masked_index_put_accumulate` destinations and the dense base add as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured duplicate-index scatter-reduce node that shares row-local layernorm reductions across masked position and vocabulary embedding-gradient accumulators with a dense base-update epilogue; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that fuses row algebra, validity masks, hidden reductions, batch-position accumulation, and direct dense base update while preserving bf16-to-f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 1024
VOCAB_ROWS = 50257
DROP_SCALE = 1.1111111111111112


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
    base_ptr,
    position_out_ptr,
    vocab_out_ptr,
    POS_TOTAL: tl.constexpr,
    VOCAB_TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)
    pos_mask = offsets < POS_TOTAL
    vocab_mask = offsets < VOCAB_TOTAL
    base = tl.load(base_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(position_out_ptr + offsets, zeros, mask=pos_mask)
    tl.store(vocab_out_ptr + offsets, base, mask=vocab_mask)


@triton.jit
def _seq_scatter_partials_kernel(
    x_bf16_ptr,
    weight_ptr,
    addend_ptr,
    broadcast_ptr,
    keep_ptr,
    mean_ptr,
    invstd_ptr,
    residual_ptr,
    position_idx_ptr,
    token_idx_ptr,
    partial_x_xhat_ptr,
    partial_x_ptr,
    position_out_ptr,
    vocab_out_ptr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seq = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    seq_offsets = seq * HIDDEN_ + cols

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    broadcast = tl.load(broadcast_ptr + seq_offsets, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_xhat = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_position = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for batch in tl.static_range(0, BATCH_):
        row = batch * SEQ_ + seq
        offsets = row * HIDDEN_ + cols
        mask = col_mask

        x = tl.load(x_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = _mul_rn(x, weight)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)

        addend = tl.load(addend_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        added = _add_rn(addend, broadcast)
        masked = _mul_rn(keep, added)
        dropped = _mul_rn(masked, DROP_SCALE_)
        mean = tl.load(mean_ptr + row, mask=True, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + row, mask=True, other=0.0).to(tl.float32)
        centered = _sub_rn(dropped, mean)
        xhat = _mul_rn(centered, invstd)

        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, xhat), 0.0), axis=0)
        scaled_weighted = _mul_rn(weighted, 768.0)
        sub1 = _sub_rn(scaled_weighted, row_sum)
        sub2 = _sub_rn(sub1, _mul_rn(xhat, row_dot))
        div = _mul_rn(invstd, 0.0013020833333333333)
        ln_grad = _mul_rn(div, sub2)

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add1 = _add_rn(residual, ln_grad)
        dropout = _mul_rn(keep, DROP_SCALE_)
        producer = _mul_rn(add1, dropout)

        acc_x_xhat += tl.where(mask, _mul_rn(x, xhat), 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_position += tl.where(mask, producer, 0.0)

        token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
        token_valid = col_mask & (token_raw >= 0) & (token_raw < VOCAB_ROWS_) & (token_raw != -1)
        tl.atomic_add(
            vocab_out_ptr + token_raw * HIDDEN_ + cols,
            producer,
            sem="relaxed",
            mask=token_valid,
        )

    position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
    position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
    position_valid = col_mask & (position_wrapped >= 0) & (position_wrapped < POSITION_ROWS_)
    tl.atomic_add(
        position_out_ptr + position_wrapped * HIDDEN_ + cols,
        acc_position,
        sem="relaxed",
        mask=position_valid,
    )

    partial_offsets = seq * HIDDEN_ + cols
    tl.store(partial_x_xhat_ptr + partial_offsets, acc_x_xhat, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)


@triton.jit
def _finalize_hidden_kernel(
    partial_x_xhat_ptr,
    partial_x_ptr,
    out_x_xhat_ptr,
    out_x_ptr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    seqs = tl.arange(0, BLOCK_SEQ)
    mask = (seqs[:, None] < SEQ_) & (cols[None, :] < HIDDEN_)
    offsets = seqs[:, None] * HIDDEN_ + cols[None, :]
    x_xhat = tl.load(partial_x_xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    col_mask = cols < HIDDEN_
    tl.store(out_x_xhat_ptr + cols, tl.sum(x_xhat, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="e66c92a5",
    INIT_BLOCK=2048,
    BLOCK_H=1024,
    FINAL_BLOCK_H=16,
    num_warps_init=4,
    num_warps_seq=8,
    num_warps_final=8,
)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
    num_warps_init: int,
    num_warps_seq: int,
    num_warps_final: int,
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

    device = arg1_1.device
    partial_x_xhat = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_x = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out0 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    pos_total = POSITION_ROWS * HIDDEN
    vocab_total = VOCAB_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(vocab_total, INIT_BLOCK),)](
        arg0_1,
        out_position,
        out_vocab,
        POS_TOTAL=pos_total,
        VOCAB_TOTAL=vocab_total,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _seq_scatter_partials_kernel[(SEQ,)](
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
        partial_x_xhat,
        partial_x,
        out_position,
        out_vocab,
        BATCH_=BATCH,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROP_SCALE_=DROP_SCALE,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_seq,
    )

    _finalize_hidden_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_H),)](
        partial_x_xhat,
        partial_x,
        out0,
        out1,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_SEQ=_ceil_pow2(SEQ),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=num_warps_final,
    )

    return out0, out1, out_position, out_vocab
