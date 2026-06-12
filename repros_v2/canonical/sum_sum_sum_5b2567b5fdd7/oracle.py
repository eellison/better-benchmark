"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 DeiT/GPT-Neo layer-norm-backward tail by reducing each hidden row once, storing the returned f32 add tensor and bf16 flattened/transpose side output, and cooperatively finalizing the two source column reductions plus the bf16-rounded side-output column sum from shared row-tile partials, whereas Inductor currently schedules the row-local reductions, dependent pointwise epilogue, bf16 materialization, transpose view, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reduction scalars live while also emitting required side outputs and compatible column partials with the explicit bf16 rounding boundary; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer/finalizer schedule for layer-norm-backward tails that stores view-equivalent side outputs and finalizes all sibling column reductions together."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _ln_bwd_store_partials_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    residual_ptr,
    add_out_ptr,
    bf16_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROW_FACTOR: tl.constexpr,
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

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight[None, :]
        weighted_rhs = weighted * rhs
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)
        scaled_weighted = weighted * ROW_FACTOR
        centered = scaled_weighted - row_sum[:, None]
        centered = centered - rhs * row_dot[:, None]
        grad = scale[:, None] * centered
        side = residual + grad
        side_bf16 = side.to(tl.bfloat16)

        tl.store(add_out_ptr + offsets, side, mask=mask)
        tl.store(bf16_out_ptr + offsets, side_bf16, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_bf16_side += tl.sum(tl.where(mask, side_bf16.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * CHANNELS + cols
    tl.store(partials_ptr + partial_base, acc_x_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + CHANNELS, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * CHANNELS, acc_bf16_side, mask=col_mask)


@triton.jit
def _ln_bwd_store_partials_c768_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    residual_ptr,
    add_out_ptr,
    bf16_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    ROW_FACTOR: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    CHUNK_C: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < 768
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_bf16_side = tl.zeros((BLOCK_C,), dtype=tl.float32)
    chunk_cols = tl.arange(0, CHUNK_C)
    acc_x_rhs0 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x_rhs1 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x_rhs2 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x0 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x1 = tl.zeros((CHUNK_C,), dtype=tl.float32)
    acc_x2 = tl.zeros((CHUNK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * 768 + cols[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        chunk_mask = row_mask[:, None]
        offsets0 = rows[:, None] * 768 + chunk_cols[None, :]
        offsets1 = offsets0 + CHUNK_C
        offsets2 = offsets1 + CHUNK_C
        w0 = tl.load(weight_ptr + chunk_cols, mask=chunk_cols < CHUNK_C, other=0.0).to(tl.float32)
        w1 = tl.load(weight_ptr + chunk_cols + CHUNK_C, mask=chunk_cols < CHUNK_C, other=0.0).to(tl.float32)
        w2 = tl.load(weight_ptr + chunk_cols + 2 * CHUNK_C, mask=chunk_cols < CHUNK_C, other=0.0).to(tl.float32)
        x0 = tl.load(x_ptr + offsets0, mask=chunk_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + offsets1, mask=chunk_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + offsets2, mask=chunk_mask, other=0.0).to(tl.float32)
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
        acc_x_rhs0 += tl.sum(_mul_rn(x0, r0), axis=0)
        acc_x_rhs1 += tl.sum(_mul_rn(x1, r1), axis=0)
        acc_x_rhs2 += tl.sum(_mul_rn(x2, r2), axis=0)
        acc_x0 += tl.sum(x0, axis=0)
        acc_x1 += tl.sum(x1, axis=0)
        acc_x2 += tl.sum(x2, axis=0)

        weighted = _mul_rn(x, weight[None, :])
        weighted_rhs = _mul_rn(weighted, rhs)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)
        row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR, tl.float32)
        scaled_weighted = _mul_rn(weighted, row_factor)
        centered = _sub_rn(scaled_weighted, row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(scale[:, None], centered)
        side = _add_rn(residual, grad)
        round_bias = _mul_rn(side, tl.full((BLOCK_R, BLOCK_C), 1.1920928955078125e-7, tl.float32))
        side_bf16 = _add_rn(side, round_bias).to(tl.bfloat16)

        tl.store(add_out_ptr + offsets, side, mask=mask)
        tl.store(bf16_out_ptr + offsets, side_bf16, mask=mask)

        acc_bf16_side += tl.sum(tl.where(mask, side_bf16.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * 768 + cols
    chunk_base = group * 3 * 768 + chunk_cols
    tl.store(partials_ptr + chunk_base, acc_x_rhs0)
    tl.store(partials_ptr + chunk_base + CHUNK_C, acc_x_rhs1)
    tl.store(partials_ptr + chunk_base + 2 * CHUNK_C, acc_x_rhs2)
    tl.store(partials_ptr + chunk_base + 768, acc_x0)
    tl.store(partials_ptr + chunk_base + 768 + CHUNK_C, acc_x1)
    tl.store(partials_ptr + chunk_base + 768 + 2 * CHUNK_C, acc_x2)
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

    side_sum = tl.sum(bf16_side, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_bf16_sum_ptr + cols, side_sum, mask=col_mask)


# 0243aeaa: (T([25216, 192], bf16), T([192], f32), ..., S([128,197,192]))
# 8348e069: (T([25344, 768], bf16), T([768], f32), ..., S([128,198,768]))
# 670b1ed7: (T([32768, 768], bf16), T([768], f32), ..., S([128,256,768]))
# 18835b6c: (T([4096, 2048], bf16), T([2048], f32), ..., S([32,128,2048]))
@oracle_impl(hardware="B200", point="0243aeaa", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=256, FINAL_BLOCK_C=8, num_warps=4)
@oracle_impl(hardware="B200", point="8348e069", ROWS_PER_GROUP=16, BLOCK_R=2, BLOCK_C=1024, FINAL_BLOCK_C=2, num_warps=8)
@oracle_impl(hardware="B200", point="670b1ed7", ROWS_PER_GROUP=16, BLOCK_R=2, BLOCK_C=1024, FINAL_BLOCK_C=2, num_warps=8)
@oracle_impl(hardware="B200", point="18835b6c", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=2048, FINAL_BLOCK_C=2, num_warps=8)
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
        x_bf16,
        weight,
        rhs,
        scale,
        residual,
        add_shape_param,
        flat_shape_param,
        sum_shape_param,
    ) = inputs
    rows = int(x_bf16.shape[0])
    channels = int(x_bf16.shape[1])
    add_shape = tuple(int(dim) for dim in add_shape_param)
    flat_shape = tuple(int(dim) for dim in flat_shape_param)
    sum_shape = tuple(int(dim) for dim in sum_shape_param)

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    out_x_rhs = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_x = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_bf16_sum = torch.empty_strided(
        sum_shape, (1,), device=x_bf16.device, dtype=torch.float32
    )

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, channels),
        (3 * channels, channels, 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )

    if channels == 768:
        _ln_bwd_store_partials_c768_kernel[(num_groups,)](
            x_bf16,
            weight,
            rhs,
            scale,
            residual,
            add_out,
            bf16_out,
            partials,
            ROWS=rows,
            ROW_FACTOR=192.0,
            ROWS_PER_GROUP=ROWS_PER_GROUP,
            BLOCK_R=BLOCK_R,
            BLOCK_C=BLOCK_C,
            CHUNK_C=256,
            num_warps=num_warps,
        )
    else:
        _ln_bwd_store_partials_kernel[(num_groups,)](
            x_bf16,
            weight,
            rhs,
            scale,
            residual,
            add_out,
            bf16_out,
            partials,
            ROWS=rows,
            CHANNELS=channels,
            ROW_FACTOR=192.0,
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
    return out_x_rhs, out_x, add_out, bf16_out, bf16_out.permute(1, 0), out_bf16_sum
