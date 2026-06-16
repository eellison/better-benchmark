"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete transformer layer-norm-backward/dropout reduction scope by row-tiling the bf16/fp32 producer, sharing each row's hidden-dimension reductions, writing the returned f32 gradient tensor plus bf16 view/transpose side output, and cooperatively finalizing all three returned hidden-column reductions from the same row-tile partials, whereas Inductor lowers the graph as separate generic row reductions, pointwise stores, view fanout, and sibling column reductions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that preserves the explicit bf16 rounding boundaries while coordinating row-local scalars, side-output stores, and compatible column partials; the fix is COOPERATIVE_SPLIT_K: add a guarded layer-norm-backward row-tiled reduction plan that keeps row summaries live, sinks view-only layout algebra into output planning, and finalizes sibling column reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROW_FACTOR = 1024.0
RETURN_DROP_SCALE = 1.1111111111111112
EXACT_DROP_SCALE = 1.109375


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
def _row_store_partials_kernel(
    x_bf16_ptr,
    residual_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    keep_ptr,
    grad_out_ptr,
    drop_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    RETURN_DROP_SCALE_: tl.constexpr,
    EXACT_DROP_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor = tl.full((BLOCK_R, BLOCK_H), ROW_FACTOR_, tl.float32)
    return_drop_scale = tl.full((BLOCK_R, BLOCK_H), RETURN_DROP_SCALE_, tl.float32)
    exact_drop_scale = tl.full((BLOCK_R, BLOCK_H), EXACT_DROP_SCALE_, tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_drop = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        offsets = rows[:, None] * HIDDEN + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = _add_rn(
            tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(x_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        )
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.int1)

        weighted = _mul_rn(x, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, rhs), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(row_scale[:, None], centered)

        returned_drop = tl.where(
            keep,
            _mul_rn(grad, return_drop_scale),
            0.0,
        ).to(tl.bfloat16)
        exact_drop = tl.where(
            keep,
            _mul_rn(grad.to(tl.bfloat16).to(tl.float32), exact_drop_scale),
            0.0,
        ).to(tl.bfloat16)

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(drop_out_ptr + offsets, returned_drop, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, _mul_rn(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_drop += tl.sum(tl.where(mask, exact_drop.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * HIDDEN + cols
    tl.store(partials_ptr + partial_base, acc_x_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN, acc_drop, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    sum_x_rhs_ptr,
    sum_x_ptr,
    sum_drop_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * HIDDEN + cols[None, :]

    x_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + HIDDEN, mask=mask, other=0.0).to(tl.float32)
    drop = tl.load(partials_ptr + offsets + 2 * HIDDEN, mask=mask, other=0.0).to(
        tl.float32
    )

    tl.store(sum_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(sum_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(
        sum_drop_ptr + cols,
        tl.sum(drop, axis=0).to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="540cb101", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, num_warps=8)
@oracle_impl(hardware="B200", point="75b78b2c", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_H=2048, FINAL_BLOCK_H=8, num_warps=8)
@oracle_impl(hardware="B200", point="9846b7f2", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, num_warps=8)
@oracle_impl(hardware="B200", point="f51dabe2", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=256, FINAL_BLOCK_H=8, num_warps=4)
@oracle_impl(hardware="B200", point="102da4dd", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, num_warps=8)
@oracle_impl(hardware="B200", point="7f0c11fc", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, num_warps=8)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
    num_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        shape_param_0,
        shape_param_1,
        shape_param_2,
    ) = inputs
    view_shape = _shape_tuple(shape_param_0)
    flat_shape = _shape_tuple(shape_param_1)
    out_shape = _shape_tuple(shape_param_2)
    rows = int(flat_shape[0])
    hidden = int(flat_shape[1])

    grad_out = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    drop_out = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_x_rhs = torch.empty_strided(out_shape, (1,), device=arg1_1.device, dtype=torch.float32)
    sum_x = torch.empty_strided(out_shape, (1,), device=arg1_1.device, dtype=torch.float32)
    sum_drop = torch.empty_strided(out_shape, (1,), device=arg1_1.device, dtype=torch.float32)

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, hidden),
        (3 * hidden, hidden, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _row_store_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        grad_out,
        drop_out,
        partials,
        ROWS=rows,
        HIDDEN=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_H=BLOCK_H,
        ROW_FACTOR_=ROW_FACTOR,
        RETURN_DROP_SCALE_=RETURN_DROP_SCALE,
        EXACT_DROP_SCALE_=EXACT_DROP_SCALE,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partials,
        sum_x_rhs,
        sum_x,
        sum_drop,
        NUM_GROUPS=num_groups,
        HIDDEN=hidden,
        GROUP_BLOCK=group_block,
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=8,
    )

    return grad_out, sum_x_rhs, sum_x, drop_out, drop_out.permute(1, 0), sum_drop
