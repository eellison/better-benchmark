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
    APPROX_SIDE_STORE: tl.constexpr,
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
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        weighted_rhs = _mul_rn(weighted, rhs)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)
        scaled_weighted = _mul_rn(
            weighted,
            tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32),
        )
        centered = _sub_rn(scaled_weighted, row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(scale[:, None], centered)
        grad = tl.where(mask, grad, 0.0)

        grad_bf16 = grad.to(tl.bfloat16)
        keep_scale = _mul_rn(
            keep,
            tl.full((BLOCK_R, BLOCK_C), DROP_SCALE_, tl.float32),
        ).to(tl.bfloat16)
        side_bf16 = _mul_rn(
            grad_bf16.to(tl.float32),
            keep_scale.to(tl.float32),
        ).to(tl.bfloat16)
        approx_side_bf16 = _mul_rn(grad, keep_scale.to(tl.float32)).to(tl.bfloat16)
        store_side = tl.where(APPROX_SIDE_STORE, approx_side_bf16, side_bf16)

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(bf16_out_ptr + offsets, store_side, mask=mask)

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
def _store_partials_c768_kernel(
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
    ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    CHUNK_C: tl.constexpr,
    APPROX_SIDE_STORE: tl.constexpr,
):
    group = tl.program_id(0)
    chunk_cols = tl.arange(0, CHUNK_C)
    w0 = tl.load(weight_ptr + chunk_cols).to(tl.float32)
    w1 = tl.load(weight_ptr + chunk_cols + CHUNK_C).to(tl.float32)
    w2 = tl.load(weight_ptr + chunk_cols + 2 * CHUNK_C).to(tl.float32)
    acc_x_rhs0 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x_rhs1 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x_rhs2 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x0 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x1 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x2 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_bf16_side0 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_bf16_side1 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_bf16_side2 = tl.zeros((CHUNK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        chunk_mask = row_mask[:, None]
        offsets0 = rows[:, None] * 768 + chunk_cols[None, :]
        offsets1 = offsets0 + CHUNK_C
        offsets2 = offsets1 + CHUNK_C

        x0 = tl.load(arg1_ptr + offsets0, mask=chunk_mask, other=0.0).to(tl.float32)
        x0 = _add_rn(x0, tl.load(arg0_ptr + offsets0, mask=chunk_mask, other=0.0).to(tl.float32))
        x0 = _add_rn(x0, tl.load(arg2_ptr + offsets0, mask=chunk_mask, other=0.0).to(tl.float32))
        x0 = _add_rn(x0, tl.load(arg3_ptr + offsets0, mask=chunk_mask, other=0.0).to(tl.float32))
        x1 = tl.load(arg1_ptr + offsets1, mask=chunk_mask, other=0.0).to(tl.float32)
        x1 = _add_rn(x1, tl.load(arg0_ptr + offsets1, mask=chunk_mask, other=0.0).to(tl.float32))
        x1 = _add_rn(x1, tl.load(arg2_ptr + offsets1, mask=chunk_mask, other=0.0).to(tl.float32))
        x1 = _add_rn(x1, tl.load(arg3_ptr + offsets1, mask=chunk_mask, other=0.0).to(tl.float32))
        x2 = tl.load(arg1_ptr + offsets2, mask=chunk_mask, other=0.0).to(tl.float32)
        x2 = _add_rn(x2, tl.load(arg0_ptr + offsets2, mask=chunk_mask, other=0.0).to(tl.float32))
        x2 = _add_rn(x2, tl.load(arg2_ptr + offsets2, mask=chunk_mask, other=0.0).to(tl.float32))
        x2 = _add_rn(x2, tl.load(arg3_ptr + offsets2, mask=chunk_mask, other=0.0).to(tl.float32))
        r0 = tl.load(rhs_ptr + offsets0, mask=chunk_mask, other=0.0).to(tl.float32)
        r1 = tl.load(rhs_ptr + offsets1, mask=chunk_mask, other=0.0).to(tl.float32)
        r2 = tl.load(rhs_ptr + offsets2, mask=chunk_mask, other=0.0).to(tl.float32)

        wx0 = _mul_rn(x0, w0[None, :])
        wx1 = _mul_rn(x1, w1[None, :])
        wx2 = _mul_rn(x2, w2[None, :])
        row_sum = _add_rn(
            _add_rn(tl.sum(wx0, axis=1), tl.sum(wx1, axis=1)),
            tl.sum(wx2, axis=1),
        )
        row_dot = _add_rn(
            _add_rn(tl.sum(_mul_rn(wx0, r0), axis=1), tl.sum(_mul_rn(wx1, r1), axis=1)),
            tl.sum(_mul_rn(wx2, r2), axis=1),
        )
        acc_x_rhs0 += tl.sum(_mul_rn(x0, r0), axis=0)
        acc_x_rhs1 += tl.sum(_mul_rn(x1, r1), axis=0)
        acc_x_rhs2 += tl.sum(_mul_rn(x2, r2), axis=0)
        acc_x0 += tl.sum(x0, axis=0)
        acc_x1 += tl.sum(x1, axis=0)
        acc_x2 += tl.sum(x2, axis=0)

        row_factor = tl.full((BLOCK_R, CHUNK_C), ROW_FACTOR_, tl.float32)
        drop_scale = tl.full((BLOCK_R, CHUNK_C), DROP_SCALE_, tl.float32)

        scaled_weighted0 = _mul_rn(wx0, row_factor)
        centered0 = _sub_rn(scaled_weighted0, row_sum[:, None])
        centered0 = _sub_rn(centered0, _mul_rn(r0, row_dot[:, None]))
        grad0 = tl.where(chunk_mask, _mul_rn(scale[:, None], centered0), 0.0)
        keep0 = tl.load(mask_ptr + offsets0, mask=chunk_mask, other=0).to(tl.float32)
        grad_bf160 = grad0.to(tl.bfloat16)
        keep_scale0 = _mul_rn(keep0, drop_scale).to(tl.bfloat16)
        side_bf160 = _mul_rn(
            grad_bf160.to(tl.float32),
            keep_scale0.to(tl.float32),
        ).to(tl.bfloat16)
        approx_side_bf160 = _mul_rn(grad0, keep_scale0.to(tl.float32)).to(tl.bfloat16)
        store_side0 = tl.where(APPROX_SIDE_STORE, approx_side_bf160, side_bf160)
        tl.store(grad_out_ptr + offsets0, grad0, mask=chunk_mask)
        tl.store(bf16_out_ptr + offsets0, store_side0, mask=chunk_mask)
        acc_bf16_side0 += tl.sum(
            tl.where(chunk_mask, side_bf160.to(tl.float32), 0.0),
            axis=0,
        )

        scaled_weighted1 = _mul_rn(wx1, row_factor)
        centered1 = _sub_rn(scaled_weighted1, row_sum[:, None])
        centered1 = _sub_rn(centered1, _mul_rn(r1, row_dot[:, None]))
        grad1 = tl.where(chunk_mask, _mul_rn(scale[:, None], centered1), 0.0)
        keep1 = tl.load(mask_ptr + offsets1, mask=chunk_mask, other=0).to(tl.float32)
        grad_bf161 = grad1.to(tl.bfloat16)
        keep_scale1 = _mul_rn(keep1, drop_scale).to(tl.bfloat16)
        side_bf161 = _mul_rn(
            grad_bf161.to(tl.float32),
            keep_scale1.to(tl.float32),
        ).to(tl.bfloat16)
        approx_side_bf161 = _mul_rn(grad1, keep_scale1.to(tl.float32)).to(tl.bfloat16)
        store_side1 = tl.where(APPROX_SIDE_STORE, approx_side_bf161, side_bf161)
        tl.store(grad_out_ptr + offsets1, grad1, mask=chunk_mask)
        tl.store(bf16_out_ptr + offsets1, store_side1, mask=chunk_mask)
        acc_bf16_side1 += tl.sum(
            tl.where(chunk_mask, side_bf161.to(tl.float32), 0.0),
            axis=0,
        )

        scaled_weighted2 = _mul_rn(wx2, row_factor)
        centered2 = _sub_rn(scaled_weighted2, row_sum[:, None])
        centered2 = _sub_rn(centered2, _mul_rn(r2, row_dot[:, None]))
        grad2 = tl.where(chunk_mask, _mul_rn(scale[:, None], centered2), 0.0)
        keep2 = tl.load(mask_ptr + offsets2, mask=chunk_mask, other=0).to(tl.float32)
        grad_bf162 = grad2.to(tl.bfloat16)
        keep_scale2 = _mul_rn(keep2, drop_scale).to(tl.bfloat16)
        side_bf162 = _mul_rn(
            grad_bf162.to(tl.float32),
            keep_scale2.to(tl.float32),
        ).to(tl.bfloat16)
        approx_side_bf162 = _mul_rn(grad2, keep_scale2.to(tl.float32)).to(tl.bfloat16)
        store_side2 = tl.where(APPROX_SIDE_STORE, approx_side_bf162, side_bf162)
        tl.store(grad_out_ptr + offsets2, grad2, mask=chunk_mask)
        tl.store(bf16_out_ptr + offsets2, store_side2, mask=chunk_mask)
        acc_bf16_side2 += tl.sum(
            tl.where(chunk_mask, side_bf162.to(tl.float32), 0.0),
            axis=0,
        )

    chunk_base = group * 3 * 768 + chunk_cols
    tl.store(partials_ptr + chunk_base, acc_x_rhs0)
    tl.store(partials_ptr + chunk_base + CHUNK_C, acc_x_rhs1)
    tl.store(partials_ptr + chunk_base + 2 * CHUNK_C, acc_x_rhs2)
    tl.store(partials_ptr + chunk_base + 768, acc_x0)
    tl.store(partials_ptr + chunk_base + 768 + CHUNK_C, acc_x1)
    tl.store(partials_ptr + chunk_base + 768 + 2 * CHUNK_C, acc_x2)
    tl.store(partials_ptr + chunk_base + 2 * 768, acc_bf16_side0)
    tl.store(partials_ptr + chunk_base + 2 * 768 + CHUNK_C, acc_bf16_side1)
    tl.store(partials_ptr + chunk_base + 2 * 768 + 2 * CHUNK_C, acc_bf16_side2)


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

    side_sum = tl.sum(bf16_side, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_bf16_sum_ptr + cols, side_sum, mask=col_mask)


# 6de23498: (T([4096,1536], bf16), T([8,512,1536], f32), ...)
# c21f4298: (T([32768,768], bf16), T([256,128,768], f32), ...)
# c1e38d67: (T([16384,768], bf16), T([32,512,768], f32), ...)
# 5acb4703: (T([32768,256], bf16), T([64,512,256], f32), ...)
@oracle_impl(hardware="B200", point="6de23498", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=2048, FINAL_BLOCK_C=2, APPROX_SIDE_STORE=False, num_warps=8)
@oracle_impl(hardware="B200", point="c21f4298", ROWS_PER_GROUP=128, BLOCK_R=4, BLOCK_C=1024, FINAL_BLOCK_C=2, APPROX_SIDE_STORE=True, num_warps=8)
@oracle_impl(hardware="B200", point="c1e38d67", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=2, APPROX_SIDE_STORE=False, num_warps=8)
@oracle_impl(hardware="B200", point="5acb4703", ROWS_PER_GROUP=64, BLOCK_R=8, BLOCK_C=256, FINAL_BLOCK_C=8, APPROX_SIDE_STORE=True, num_warps=4)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    APPROX_SIDE_STORE: bool,
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

    if channels == 768:
        _store_partials_c768_kernel[(num_groups,)](
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
            ROW_FACTOR_=ROW_FACTOR,
            DROP_SCALE_=DROP_SCALE,
            ROWS_PER_GROUP=ROWS_PER_GROUP,
            BLOCK_R=BLOCK_R,
            BLOCK_C=BLOCK_C,
            CHUNK_C=256,
            APPROX_SIDE_STORE=APPROX_SIDE_STORE,
            num_warps=num_warps,
        )
    else:
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
            APPROX_SIDE_STORE=APPROX_SIDE_STORE,
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
