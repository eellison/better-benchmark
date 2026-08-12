"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GPT2 sequence-classification layernorm-backward plus embedding-gradient scatter scope by reconstructing the bf16-to-fp32 hidden producer, row-local 768-wide reductions, dropout-mask arithmetic, two sibling `[768]` column reductions, the returned `[1024,768]` position scatter, and the returned `[50257,768]` vocabulary scatter in Triton, whereas Inductor lowers the row reductions, dense pointwise epilogue, zero fills, batch-position reduction, and duplicate-preserving `_unsafe_masked_index_put_accumulate` outputs as separate generic schedules over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no scatter-reduce template that fuses a row-local layernorm-backward producer with multiple indexed accumulation destinations while preserving the captured f32 arithmetic, masks, full dense output scope, and duplicate-safe atomics; the fix is SCATTER_REDUCE: add a guarded embedding-gradient scatter-reduce lowering that initializes dense destinations directly, folds validity masks into indexed updates, shares the row producer across scatter outputs, and finalizes compatible hidden-column reductions from cooperative row partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768
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
    pos_out_ptr,
    vocab_out_ptr,
    POS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), tl.float32)
    tl.store(pos_out_ptr + offsets, zeros, mask=offsets < POS_N)
    tl.store(vocab_out_ptr + offsets, zeros, mask=offsets < VOCAB_N)


@triton.jit
def _row_scatter_partials_kernel(
    x_bf16_ptr,
    weight_ptr,
    addend_ptr,
    token_addend_ptr,
    keep_ptr,
    sub_ptr,
    scale_ptr,
    residual_ptr,
    pos_index_ptr,
    vocab_index_ptr,
    partial_x_grad_ptr,
    partial_x_ptr,
    pos_out_ptr,
    vocab_out_ptr,
    ROWS_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    INV_HIDDEN_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_grad = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for row_offset in tl.static_range(0, ROWS_PER_GROUP):
        row = group * ROWS_PER_GROUP + row_offset
        row_active = row < ROWS_
        token = row - (row // SEQ_) * SEQ_
        offsets = row * HIDDEN_ + cols
        token_offsets = token * HIDDEN_ + cols
        mask = row_active & col_mask

        x = tl.load(x_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = _mul_rn(x, weight)
        weighted_scaled = _mul_rn(weighted, ROW_FACTOR_)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)

        addend = tl.load(addend_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        token_addend = tl.load(token_addend_ptr + token_offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        add = _add_rn(addend, token_addend)
        dropped = _mul_rn(_mul_rn(keep, add), DROP_SCALE_)
        centered = _sub_rn(
            dropped,
            tl.load(sub_ptr + row, mask=row_active, other=0.0).to(tl.float32),
        )
        row_grad = _mul_rn(
            centered,
            tl.load(scale_ptr + row, mask=row_active, other=0.0).to(tl.float32),
        )

        weighted_grad = _mul_rn(weighted, row_grad)
        row_dot = tl.sum(tl.where(mask, weighted_grad, 0.0), axis=0)
        centered_weighted = _sub_rn(_sub_rn(weighted_scaled, row_sum), _mul_rn(row_grad, row_dot))
        inv_scale = _mul_rn(
            tl.load(scale_ptr + row, mask=row_active, other=0.0).to(tl.float32),
            INV_HIDDEN_,
        )
        ln_grad = _mul_rn(inv_scale, centered_weighted)

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add_1 = _add_rn(residual, ln_grad)
        scatter_value = _mul_rn(add_1, _mul_rn(keep, DROP_SCALE_))

        pos_raw = tl.load(pos_index_ptr + token, mask=row_active, other=0).to(tl.int64)
        pos_row = tl.where(pos_raw < 0, pos_raw + POSITION_ROWS_, pos_raw)
        pos_active = mask & (pos_row >= 0) & (pos_row < POSITION_ROWS_)
        tl.atomic_add(
            pos_out_ptr + pos_row * HIDDEN_ + cols,
            scatter_value,
            sem="relaxed",
            mask=pos_active,
        )

        vocab_row = tl.load(vocab_index_ptr + row, mask=row_active, other=-1).to(tl.int64)
        vocab_active = mask & (vocab_row >= 0) & (vocab_row < VOCAB_ROWS_) & (vocab_row != -1)
        tl.atomic_add(
            vocab_out_ptr + vocab_row * HIDDEN_ + cols,
            scatter_value,
            sem="relaxed",
            mask=vocab_active,
        )

        acc_x_grad += tl.where(mask, _mul_rn(x, row_grad), 0.0)
        acc_x += tl.where(mask, x, 0.0)

    partial_offsets = group * HIDDEN_ + cols
    tl.store(partial_x_grad_ptr + partial_offsets, acc_x_grad, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_x_grad_ptr,
    partial_x_ptr,
    out_x_grad_ptr,
    out_x_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * HIDDEN_ + cols[None, :]

    x_grad = tl.load(partial_x_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_grad_ptr + cols, tl.sum(x_grad, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="7e991e0f",
    ROWS_PER_GROUP=8,
    BLOCK_H=1024,
    FINAL_BLOCK_H=16,
    INIT_BLOCK=2048,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
    INIT_BLOCK: int,
    row_warps: int,
    final_warps: int,
):
    (
        x_bf16,
        weight,
        addend,
        token_addend,
        keep,
        sub,
        scale,
        residual,
        pos_index,
        vocab_index,
        *_shape_params,
    ) = inputs

    device = x_bf16.device
    hidden = int(weight.numel())
    rows = int(x_bf16.numel() // hidden)
    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)

    partial_x_grad = torch.empty((num_groups, hidden), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_groups, hidden), device=device, dtype=torch.float32)
    out_x_grad = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_x = torch.empty((hidden,), device=device, dtype=torch.float32)
    pos_out = torch.empty((POSITION_ROWS, hidden), device=device, dtype=torch.float32)
    vocab_out = torch.empty((VOCAB_ROWS, hidden), device=device, dtype=torch.float32)

    pos_n = POSITION_ROWS * hidden
    vocab_n = VOCAB_ROWS * hidden
    _init_outputs_kernel[(triton.cdiv(vocab_n, INIT_BLOCK),)](
        pos_out,
        vocab_out,
        POS_N=pos_n,
        VOCAB_N=vocab_n,
        BLOCK=INIT_BLOCK,
        num_warps=4,
    )
    _row_scatter_partials_kernel[(num_groups,)](
        x_bf16,
        weight,
        addend,
        token_addend,
        keep,
        sub,
        scale,
        residual,
        pos_index,
        vocab_index,
        partial_x_grad,
        partial_x,
        pos_out,
        vocab_out,
        ROWS_=rows,
        SEQ_=SEQ,
        HIDDEN_=hidden,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_H=BLOCK_H,
        DROP_SCALE_=DROP_SCALE,
        ROW_FACTOR_=float(HIDDEN),
        INV_HIDDEN_=1.0 / HIDDEN,
        num_warps=row_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partial_x_grad,
        partial_x,
        out_x_grad,
        out_x,
        NUM_GROUPS=num_groups,
        HIDDEN_=hidden,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=final_warps,
    )

    return out_x_grad, out_x, pos_out, vocab_out
