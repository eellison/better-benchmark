"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16/fp32 masked-LM layer-norm-backward tail by rebuilding the four-input producer, sharing each row's hidden-dimension reductions, storing the returned fp32 gradient and bf16 dropout-masked side tensor, and cooperatively finalizing the two source column reductions plus the bf16-rounded side-output sum from common row-tile partials, whereas Inductor currently schedules the add chain, row-local reductions, dropout-mask/bf16 epilogue, transpose view, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, explicit bf16 rounding boundaries, layout side outputs, and compatible column partials in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled layer-norm-backward reduction template that fuses shared producers, writes view-equivalent side outputs, and finalizes all sibling column reductions together."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROW_FACTOR = 1536.0
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
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    mask_ptr,
    grad_out_ptr,
    bf16_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_bf16_side = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * CHANNELS + cols[None, :]

        x = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = _add_rn(x, tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        x = _add_rn(x, tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        x = _add_rn(x, tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        keep = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.float32)

        weighted = _mul_rn(x, weight[None, :])
        weighted_rhs = _mul_rn(weighted, rhs)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0).to(tl.float64), axis=1).to(tl.float32)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0).to(tl.float64), axis=1).to(tl.float32)
        scaled_weighted = _mul_rn(
            weighted,
            tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32),
        )
        centered = _sub_rn(scaled_weighted, row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(scale[:, None], centered)
        grad = tl.where(mask, grad, 0.0)

        grad_bf16 = _round_bf16_to_f32(grad)
        keep_scale = _round_bf16_to_f32(_mul_rn(
            keep,
            tl.full((BLOCK_R, BLOCK_C), DROP_SCALE_, tl.float32),
        ))
        side_bf16 = _round_bf16_to_f32(_mul_rn(grad_bf16, keep_scale))

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(bf16_out_ptr + offsets, side_bf16, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, _mul_rn(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_bf16_side += tl.sum(
            tl.where(mask, side_bf16.to(tl.float32), 0.0),
            axis=0,
        )

    partial_base = group * 3 * CHANNELS + cols
    tl.store(partials_ptr + partial_base, acc_x_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + CHANNELS, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * CHANNELS, acc_bf16_side, mask=col_mask)


@triton.jit
def _c768_tree_row_store_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    mask_ptr,
    grad_out_ptr,
    bf16_out_ptr,
    ROWS: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    STORE_BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0) * STORE_BLOCK_R + tl.arange(0, STORE_BLOCK_R)
    row_mask = row < ROWS
    lanes = tl.arange(0, 32)
    sum0 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)
    sum1 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)
    sum2 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)
    sum3 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)
    dot0 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)
    dot1 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)
    dot2 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)
    dot3 = tl.zeros((STORE_BLOCK_R, 32), dtype=tl.float32)

    for step in tl.static_range(0, 6):
        for vec in tl.static_range(0, 4):
            col = lanes * 4 + step * 128 + vec
            offs = row[:, None] * 768 + col[None, :]
            tree_mask = row_mask[:, None]
            x_tree = tl.load(arg1_ptr + offs, mask=tree_mask, other=0.0).to(tl.float32)
            x_tree = _add_rn(
                x_tree,
                tl.load(arg0_ptr + offs, mask=tree_mask, other=0.0).to(tl.float32),
            )
            x_tree = _add_rn(
                x_tree,
                tl.load(arg2_ptr + offs, mask=tree_mask, other=0.0).to(tl.float32),
            )
            x_tree = _add_rn(
                x_tree,
                tl.load(arg3_ptr + offs, mask=tree_mask, other=0.0).to(tl.float32),
            )
            rhs_tree = tl.load(rhs_ptr + offs, mask=tree_mask, other=0.0).to(tl.float32)
            weighted_tree = _mul_rn(
                x_tree,
                tl.load(weight_ptr + col).to(tl.float32)[None, :],
            )
            weighted_rhs_tree = _mul_rn(weighted_tree, rhs_tree)
            if vec == 0:
                sum0 = _add_rn(sum0, weighted_tree)
                dot0 = _add_rn(dot0, weighted_rhs_tree)
            elif vec == 1:
                sum1 = _add_rn(sum1, weighted_tree)
                dot1 = _add_rn(dot1, weighted_rhs_tree)
            elif vec == 2:
                sum2 = _add_rn(sum2, weighted_tree)
                dot2 = _add_rn(dot2, weighted_rhs_tree)
            else:
                sum3 = _add_rn(sum3, weighted_tree)
                dot3 = _add_rn(dot3, weighted_rhs_tree)

    lane_sum = _add_rn(_add_rn(_add_rn(sum0, sum1), sum2), sum3)
    lane_dot = _add_rn(_add_rn(_add_rn(dot0, dot1), dot2), dot3)
    sum16 = tl.reshape(tl.sum(tl.reshape(lane_sum, (STORE_BLOCK_R * 16, 2)), axis=1), (STORE_BLOCK_R, 16))
    dot16 = tl.reshape(tl.sum(tl.reshape(lane_dot, (STORE_BLOCK_R * 16, 2)), axis=1), (STORE_BLOCK_R, 16))
    sum8 = tl.reshape(tl.sum(tl.reshape(sum16, (STORE_BLOCK_R * 8, 2)), axis=1), (STORE_BLOCK_R, 8))
    dot8 = tl.reshape(tl.sum(tl.reshape(dot16, (STORE_BLOCK_R * 8, 2)), axis=1), (STORE_BLOCK_R, 8))
    sum4 = tl.reshape(tl.sum(tl.reshape(sum8, (STORE_BLOCK_R * 4, 2)), axis=1), (STORE_BLOCK_R, 4))
    dot4 = tl.reshape(tl.sum(tl.reshape(dot8, (STORE_BLOCK_R * 4, 2)), axis=1), (STORE_BLOCK_R, 4))
    sum2_final = tl.reshape(tl.sum(tl.reshape(sum4, (STORE_BLOCK_R * 2, 2)), axis=1), (STORE_BLOCK_R, 2))
    dot2_final = tl.reshape(tl.sum(tl.reshape(dot4, (STORE_BLOCK_R * 2, 2)), axis=1), (STORE_BLOCK_R, 2))
    row_sum = tl.sum(sum2_final, axis=1)
    row_dot = tl.sum(dot2_final, axis=1)

    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < 768
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row[:, None] * 768 + cols[None, :]
    x = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = _add_rn(x, tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
    x = _add_rn(x, tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
    x = _add_rn(x, tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
    keep = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    weighted = _mul_rn(x, weight[None, :])
    scaled_weighted = _mul_rn(
        weighted,
        tl.full((STORE_BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32),
    )
    centered = _sub_rn(scaled_weighted, row_sum[:, None])
    centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
    grad = _mul_rn(scale[:, None], centered)
    grad = tl.where(mask, grad, 0.0)

    grad_bf16 = _round_bf16_to_f32(grad)
    keep_scale = _round_bf16_to_f32(_mul_rn(
        keep,
        tl.full((STORE_BLOCK_R, BLOCK_C), DROP_SCALE_, tl.float32),
    ))
    side_bf16 = _round_bf16_to_f32(_mul_rn(grad_bf16, keep_scale))
    tl.store(grad_out_ptr + offsets, grad, mask=mask)
    tl.store(bf16_out_ptr + offsets, side_bf16, mask=mask)


@triton.jit
def _c21_column_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    rhs_ptr,
    bf16_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < 768
    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_bf16_side = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * 768 + cols[None, :]
        x = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = _add_rn(x, tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        x = _add_rn(x, tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        x = _add_rn(x, tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bf16_side = tl.load(bf16_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc_x_rhs += tl.sum(tl.where(mask, _mul_rn(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_bf16_side += tl.sum(tl.where(mask, bf16_side, 0.0), axis=0)

    partial_base = group * 3 * 768 + cols
    tl.store(partials_ptr + partial_base, acc_x_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + 768, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * 768, acc_bf16_side, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_bf16_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < CHANNELS)
    offsets = groups[:, None] * 3 * CHANNELS + cols[None, :]
    col_mask = cols < CHANNELS

    x_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + CHANNELS, mask=mask, other=0.0).to(tl.float32)
    bf16_side = tl.load(partials_ptr + offsets + 2 * CHANNELS, mask=mask, other=0.0).to(
        tl.float32
    )

    side_sum = tl.sum(bf16_side, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_bf16_sum_ptr + cols, side_sum, mask=col_mask)


@oracle_impl(hardware="B200", point="c21f4298", ROWS_PER_GROUP=128, BLOCK_R=4, STORE_BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=2, num_warps=2)
@oracle_impl(hardware="B200", point="c1e38d67", ROWS_PER_GROUP=64, BLOCK_R=4, STORE_BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=2, num_warps=4)
def _oracle_forward_c21(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    STORE_BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        rhs,
        scale,
        mask,
        _shape0,
        _shape1,
        _shape2,
        flat_shape_param,
        sum_shape_param,
    ) = inputs
    rows = int(arg0.shape[0])
    full_shape = tuple(int(dim) for dim in arg1.shape)
    flat_shape = tuple(int(dim) for dim in flat_shape_param)
    sum_shape = tuple(int(dim) for dim in sum_shape_param)
    device = arg1.device

    grad_out = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_x_rhs = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    out_x = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    out_bf16_sum = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, 768),
        (3 * 768, 768, 1),
        device=device,
        dtype=torch.float32,
    )
    _c768_tree_row_store_kernel[(triton.cdiv(rows, STORE_BLOCK_R),)](
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        rhs,
        scale,
        mask,
        grad_out,
        bf16_out,
        ROWS=rows,
        ROW_FACTOR_=ROW_FACTOR,
        DROP_SCALE_=DROP_SCALE,
        STORE_BLOCK_R=STORE_BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _c21_column_partials_kernel[(num_groups,)](
        arg0,
        arg1,
        arg2,
        arg3,
        rhs,
        bf16_out,
        partials,
        ROWS=rows,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(768, FINAL_BLOCK_C),)](
        partials,
        out_x_rhs,
        out_x,
        out_bf16_sum,
        NUM_GROUPS=num_groups,
        CHANNELS=768,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )
    return grad_out, out_x_rhs, out_x, bf16_out, bf16_out.permute(1, 0), out_bf16_sum


# 6de23498: (T([4096,1536], bf16), T([8,512,1536], f32), ...)
# c21f4298: (T([32768,768], bf16), T([256,128,768], f32), ...)
# 5acb4703: (T([32768,256], bf16), T([64,512,256], f32), ...)
@oracle_impl(hardware="B200", point="6de23498", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=2048, FINAL_BLOCK_C=2, num_warps=8)
@oracle_impl(hardware="B200", point="5acb4703", ROWS_PER_GROUP=64, BLOCK_R=8, BLOCK_C=256, FINAL_BLOCK_C=8, num_warps=4)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        rhs,
        scale,
        mask,
        _shape0,
        _shape1,
        _shape2,
        flat_shape_param,
        sum_shape_param,
    ) = inputs
    rows = int(arg0.shape[0])
    channels = int(arg0.shape[1])
    full_shape = tuple(int(dim) for dim in arg1.shape)
    flat_shape = tuple(int(dim) for dim in flat_shape_param)
    sum_shape = tuple(int(dim) for dim in sum_shape_param)

    grad_out = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=arg1.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=arg1.device,
        dtype=torch.bfloat16,
    )
    out_x_rhs = torch.empty_strided(sum_shape, (1,), device=arg1.device, dtype=torch.float32)
    out_x = torch.empty_strided(sum_shape, (1,), device=arg1.device, dtype=torch.float32)
    out_bf16_sum = torch.empty_strided(
        sum_shape,
        (1,),
        device=arg1.device,
        dtype=torch.float32,
    )

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, channels),
        (3 * channels, channels, 1),
        device=arg1.device,
        dtype=torch.float32,
    )

    _store_partials_kernel[(num_groups,)](
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        rhs,
        scale,
        mask,
        grad_out,
        bf16_out,
        partials,
        ROWS=rows,
        CHANNELS=channels,
        ROW_FACTOR_=ROW_FACTOR,
        DROP_SCALE_=DROP_SCALE,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
        partials,
        out_x_rhs,
        out_x,
        out_bf16_sum,
        NUM_GROUPS=num_groups,
        CHANNELS=channels,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )
    return grad_out, out_x_rhs, out_x, bf16_out, bf16_out.permute(1, 0), out_bf16_sum
