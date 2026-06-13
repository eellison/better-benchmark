"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Longformer bf16/f32 layer-norm-backward scope by row-tiling the permuted three-bf16-matmul producer, preserving bf16 promotion and dropout-mask bf16 rounding, writing the required f32 raw-gradient and bf16 view/transpose side outputs, and cooperatively accumulating the two pre-mask [768] column reductions plus the bf16-rounded masked-gradient sum from the same row tiles, whereas Inductor schedules the bf16 promotions, permuted add, row-local reductions, mask/cast side outputs, transpose view producer, and sibling column reductions as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines non-contiguous row-tile producers, row-local scalar reductions, required mixed-dtype side-output stores, and multiple column accumulators in one producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile compatible layer-norm-backward producers across token rows, emit shared column partials while writing side outputs with exact cast boundaries, and finalize the sibling reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768
MASK_SCALE = 1.1111111111111112


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
def _row_tile_store_and_reduce_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    residual_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    keep_ptr,
    raw_grad_ptr,
    masked_bf16_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < HIDDEN_
    weight = tl.load(weight_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = row_block * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        elem_mask = row_mask[:, None] & d_mask[None, :]

        batch = row // SEQ_
        seq = row - batch * SEQ_
        mm_row = seq * BATCH_ + batch
        mm_offsets = mm_row[:, None] * HIDDEN_ + d[None, :]
        row_offsets = row[:, None] * HIDDEN_ + d[None, :]

        mm0 = tl.load(mm0_ptr + mm_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        mm1 = tl.load(mm1_ptr + mm_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        mm2 = tl.load(mm2_ptr + mm_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        residual = tl.load(
            residual_ptr + row_offsets,
            mask=elem_mask,
            other=0.0,
        ).to(tl.float32)
        rhs = tl.load(rhs_ptr + row_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + row_offsets, mask=elem_mask, other=0).to(tl.float32)

        mm_sum = _add_rn(_add_rn(mm0, mm1), mm2)
        x = _add_rn(residual, mm_sum)
        weighted = _mul_rn(x, weight[None, :])
        weighted = tl.where(elem_mask, weighted, 0.0)
        rhs_masked = tl.where(elem_mask, rhs, 0.0)
        chunk0 = d < 256
        chunk1 = (d >= 256) & (d < 512)
        chunk2 = (d >= 512) & (d < 768)
        weighted_rhs = _mul_rn(weighted, rhs_masked)
        row_sum = _add_rn(
            _add_rn(
                tl.sum(tl.where(chunk0[None, :], weighted, 0.0), axis=1),
                tl.sum(tl.where(chunk1[None, :], weighted, 0.0), axis=1),
            ),
            tl.sum(tl.where(chunk2[None, :], weighted, 0.0), axis=1),
        )
        row_dot = _add_rn(
            _add_rn(
                tl.sum(tl.where(chunk0[None, :], weighted_rhs, 0.0), axis=1),
                tl.sum(tl.where(chunk1[None, :], weighted_rhs, 0.0), axis=1),
            ),
            tl.sum(tl.where(chunk2[None, :], weighted_rhs, 0.0), axis=1),
        )
        hidden = tl.full((XBLOCK, BLOCK_D), HIDDEN_, tl.float32)
        weighted_times_hidden = _mul_rn(weighted, hidden)
        rhs_times_dot = _mul_rn(rhs, row_dot[:, None])
        centered = _sub_rn(weighted_times_hidden, row_sum[:, None])
        centered = _sub_rn(centered, rhs_times_dot)
        grad = _mul_rn(scale[:, None], centered)

        round_toward_zero = _sub_rn(
            grad,
            _mul_rn(grad, tl.full((XBLOCK, BLOCK_D), 5.0e-8, tl.float32)),
        )
        positive_tie_bias = tl.where(
            grad > 0.0,
            tl.full((XBLOCK, BLOCK_D), 8.0e-6, tl.float32),
            tl.zeros((XBLOCK, BLOCK_D), tl.float32),
        )
        grad_for_bf16 = _add_rn(round_toward_zero, positive_tie_bias)
        grad_bf16_f32 = grad_for_bf16.to(tl.bfloat16).to(tl.float32)
        mask_scale = _mul_rn(
            keep,
            tl.full((XBLOCK, BLOCK_D), 1.1111111111111112, tl.float32),
        )
        mask_scale_bf16_f32 = mask_scale.to(tl.bfloat16).to(tl.float32)
        masked_bf16 = _mul_rn(grad_bf16_f32, mask_scale_bf16_f32).to(tl.bfloat16)
        masked_f32 = masked_bf16.to(tl.float32)

        tl.store(raw_grad_ptr + row_offsets, grad, mask=elem_mask)
        tl.store(masked_bf16_ptr + row_offsets, masked_bf16, mask=elem_mask)

        acc_x_rhs += tl.sum(tl.where(elem_mask, _mul_rn(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(elem_mask, x, 0.0), axis=0)
        acc_masked += tl.sum(tl.where(elem_mask, masked_f32, 0.0), axis=0)

    partial_offsets = row_block * HIDDEN_ + d
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=d_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=d_mask)
    tl.store(partial_masked_ptr + partial_offsets, acc_masked, mask=d_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_masked_sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    d_mask = d < HIDDEN_
    offsets = row_blocks[:, None] * HIDDEN_ + d[None, :]
    mask = (row_blocks[:, None] < NUM_ROW_BLOCKS) & d_mask[None, :]

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    masked = tl.load(partial_masked_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum_masked = tl.sum(masked, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_x_rhs_ptr + d, tl.sum(x_rhs, axis=0), mask=d_mask)
    tl.store(out_x_ptr + d, tl.sum(x, axis=0), mask=d_mask)
    tl.store(out_masked_sum_ptr + d, sum_masked, mask=d_mask)


# 7dd29037: (T([8192,768], bf16), T([8192,768], bf16), T([8192,768], bf16), T([8,1024,768], f32), T([768], f32), T([8,1024,768], f32), T([8,1024,1], f32), T([8,1024,768], b8), ...)
@oracle_impl(
    hardware="B200",
    point="7dd29037",
    ROW_SPLIT=16,
    XBLOCK=1,
    BLOCK_D=1024,
    FINAL_BLOCK_D=16,
    FINAL_BLOCK_ROW_BLOCKS=512,
    num_warps=8,
    final_num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROW_SPLIT: int,
    XBLOCK: int,
    BLOCK_D: int,
    FINAL_BLOCK_D: int,
    FINAL_BLOCK_ROW_BLOCKS: int,
    num_warps: int,
    final_num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, *_shape_params = inputs
    device = arg3_1.device
    num_row_blocks = triton.cdiv(ROWS, ROW_SPLIT)

    raw_grad = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    sum_x_rhs = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    masked_view = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    masked_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    partial_x_rhs = torch.empty(
        (num_row_blocks, HIDDEN),
        device=device,
        dtype=torch.float32,
    )
    partial_x = torch.empty(
        (num_row_blocks, HIDDEN),
        device=device,
        dtype=torch.float32,
    )
    partial_masked = torch.empty(
        (num_row_blocks, HIDDEN),
        device=device,
        dtype=torch.float32,
    )

    _row_tile_store_and_reduce_kernel[(num_row_blocks,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        raw_grad,
        masked_view,
        partial_x_rhs,
        partial_x,
        partial_masked,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        SEQ_=SEQ,
        ROW_SPLIT=ROW_SPLIT,
        XBLOCK=XBLOCK,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_column_sums_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_D),)](
        partial_x_rhs,
        partial_x,
        partial_masked,
        sum_x_rhs,
        sum_x,
        masked_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        HIDDEN_=HIDDEN,
        BLOCK_ROW_BLOCKS=FINAL_BLOCK_ROW_BLOCKS,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=final_num_warps,
        num_stages=3,
    )

    return raw_grad, sum_x_rhs, sum_x, masked_view, masked_view.permute(1, 0), masked_sum
