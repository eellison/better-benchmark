"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete VisFormer double-normalization backward scope, including the first channel reductions, batch-sum side output, bf16-rounded full tensor, and final bf16 reduction, by reusing compact per-channel summaries across the dependent formulas, whereas Inductor schedules the decomposed sibling and dependent reductions as generic regions over large intermediates; Inductor cannot do this today because its algebraic simplifier and reduction scheduler do not recognize this BN-backward-style chain as reusable summaries once the rounded full-tensor and side reductions are also live; the fix is ALGEBRAIC_ELIMINATION: add a guarded multi-output reduction rewrite that preserves shared channel summaries, lowers the side-output reductions directly, and materializes the rounded epilogue only at required cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
INV_REDUCE = 9.964923469387754e-06


@triton.jit
def _first_partials_kernel(
    x_ptr,
    y_ptr,
    mean1_ptr,
    inv1_ptr,
    weight1_ptr,
    bias1_ptr,
    spatial_bias_ptr,
    mean2_ptr,
    partials_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_R: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    n = r // HW
    hw = r - n * HW
    h = hw // W
    w = hw - h * W
    mask = (c[None, :] < C) & (r[:, None] < TOTAL_R)
    offsets = n[:, None] * C * HW + h[:, None] * W * C + w[:, None] * C + c[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    inv1 = tl.load(inv1_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    spatial = tl.load(
        spatial_bias_ptr + h[:, None] * W * C + w[:, None] * C + c[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    affine = ((y - mean1[None, :]) * inv1[None, :]) * weight1[None, :] + bias1[None, :]
    norm = affine.to(tl.bfloat16).to(tl.float32) + spatial - mean2[None, :]
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_x_norm = tl.sum(tl.where(mask, x * norm, 0.0), axis=0)

    tile = tl.program_id(1)
    base = c * NUM_TILES + tile
    plane = C * NUM_TILES
    tl.store(partials_ptr + base, sum_x, mask=c < C)
    tl.store(partials_ptr + plane + base, sum_x_norm, mask=c < C)


@triton.jit
def _stats1_kernel(
    partials_ptr,
    gamma2_ptr,
    beta2_ptr,
    out0_ptr,
    out1_ptr,
    stats1_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    INV_REDUCE_VALUE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (c[:, None] < C) & (tiles[None, :] < NUM_TILES)
    base = c[:, None] * NUM_TILES + tiles[None, :]
    plane = C * NUM_TILES
    sum_x_parts = tl.load(partials_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum_x_norm_parts = tl.load(partials_ptr + plane + base, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(sum_x_parts, axis=1)
    sum_x_norm = tl.sum(sum_x_norm_parts, axis=1)
    gamma2 = tl.load(gamma2_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    beta2 = tl.load(beta2_ptr + c, mask=c < C, other=0.0).to(tl.float32)

    mean_x = sum_x * INV_REDUCE_VALUE
    norm_scale = (sum_x_norm * INV_REDUCE_VALUE) * (gamma2 * gamma2)
    grad_scale = gamma2 * beta2
    tl.store(out0_ptr + c, sum_x, mask=c < C)
    tl.store(out1_ptr + c, sum_x_norm * gamma2, mask=c < C)
    tl.store(stats1_ptr + c, mean_x, mask=c < C)
    tl.store(stats1_ptr + C + c, norm_scale, mask=c < C)
    tl.store(stats1_ptr + 2 * C + c, grad_scale, mask=c < C)


@triton.jit
def _phase2_kernel(
    x_ptr,
    y_ptr,
    mean1_ptr,
    inv1_ptr,
    weight1_ptr,
    bias1_ptr,
    spatial_bias_ptr,
    mean2_ptr,
    residual_ptr,
    stats1_ptr,
    out2_ptr,
    add2_bf16_ptr,
    second_partials_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    HW_TILES: tl.constexpr,
    RESIDUAL_CHANNELS_LAST: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_base = tl.program_id(1) * BLOCK_HW
    n = tl.arange(0, 128)
    c_mask = c < C

    mean1 = tl.load(mean1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    inv1 = tl.load(inv1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    mean_x = tl.load(stats1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    norm_scale = tl.load(stats1_ptr + C + c, mask=c_mask, other=0.0).to(tl.float32)
    grad_scale = tl.load(stats1_ptr + 2 * C + c, mask=c_mask, other=0.0).to(tl.float32)

    sum4_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum5_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for hw_i in tl.static_range(0, BLOCK_HW):
        hw = hw_base + hw_i
        h = hw // W
        w = hw - h * W
        active_c = c_mask & (hw < HW)
        cl_offsets = n[:, None] * C * HW + h * W * C + w * C + c[None, :]
        if RESIDUAL_CHANNELS_LAST:
            out_offsets = cl_offsets
        else:
            out_offsets = n[:, None] * C * HW + c[None, :] * HW + hw
        mask = active_c[None, :]

        x = tl.load(x_ptr + cl_offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + cl_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
        spatial = tl.load(
            spatial_bias_ptr + h * W * C + w * C + c,
            mask=active_c,
            other=0.0,
        ).to(tl.float32)

        affine = ((y - mean1[None, :]) * inv1[None, :]) * weight1[None, :] + bias1[None, :]
        norm = affine.to(tl.bfloat16).to(tl.float32) + spatial[None, :] - mean2[None, :]
        add2 = residual + (x - norm * norm_scale[None, :] - mean_x[None, :]) * grad_scale[None, :]
        add2_bf16 = add2.to(tl.bfloat16)
        add2_f32 = add2_bf16.to(tl.float32)
        sub4 = y - mean1[None, :]

        tl.store(add2_bf16_ptr + out_offsets, add2_bf16, mask=mask)
        tl.store(out2_ptr + c * HW + hw, tl.sum(tl.where(mask, add2, 0.0), axis=0), mask=active_c)
        sum4_acc += tl.sum(tl.where(mask, add2_f32, 0.0), axis=0)
        sum5_acc += tl.sum(tl.where(mask, add2_f32 * sub4, 0.0), axis=0)

    tile = tl.program_id(1)
    base = c * HW_TILES + tile
    plane = C * HW_TILES
    tl.store(second_partials_ptr + base, sum4_acc, mask=c_mask)
    tl.store(second_partials_ptr + plane + base, sum5_acc, mask=c_mask)


@triton.jit
def _stats2_kernel(
    second_partials_ptr,
    inv1_ptr,
    weight1_ptr,
    out3_ptr,
    out4_ptr,
    stats2_ptr,
    C: tl.constexpr,
    HW_TILES: tl.constexpr,
    INV_REDUCE_VALUE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (c[:, None] < C) & (tiles[None, :] < HW_TILES)
    base = c[:, None] * HW_TILES + tiles[None, :]
    plane = C * HW_TILES
    sum4_parts = tl.load(second_partials_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum5_parts = tl.load(second_partials_ptr + plane + base, mask=mask, other=0.0).to(tl.float32)
    sum4 = tl.sum(sum4_parts, axis=1)
    sum5 = tl.sum(sum5_parts, axis=1)
    inv1 = tl.load(inv1_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=c < C, other=0.0).to(tl.float32)

    mean_out = sum4 * INV_REDUCE_VALUE
    correction = (sum5 * INV_REDUCE_VALUE) * (inv1 * inv1)
    out_scale = inv1 * weight1
    tl.store(out3_ptr + c, sum4, mask=c < C)
    tl.store(out4_ptr + c, sum5 * inv1, mask=c < C)
    tl.store(stats2_ptr + c, mean_out, mask=c < C)
    tl.store(stats2_ptr + C + c, correction, mask=c < C)
    tl.store(stats2_ptr + 2 * C + c, out_scale, mask=c < C)


@triton.jit
def _phase3_kernel(
    y_ptr,
    mean1_ptr,
    add2_bf16_ptr,
    stats2_ptr,
    out5_ptr,
    partial6_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    HW_TILES: tl.constexpr,
    OUTPUT_CHANNELS_LAST: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_base = tl.program_id(1) * BLOCK_HW
    n = tl.arange(0, 128)
    c_mask = c < C

    mean1 = tl.load(mean1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    mean_out = tl.load(stats2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    correction = tl.load(stats2_ptr + C + c, mask=c_mask, other=0.0).to(tl.float32)
    out_scale = tl.load(stats2_ptr + 2 * C + c, mask=c_mask, other=0.0).to(tl.float32)
    sum6_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for hw_i in tl.static_range(0, BLOCK_HW):
        hw = hw_base + hw_i
        h = hw // W
        w = hw - h * W
        active_c = c_mask & (hw < HW)
        cl_offsets = n[:, None] * C * HW + h * W * C + w * C + c[None, :]
        if OUTPUT_CHANNELS_LAST:
            out_offsets = cl_offsets
        else:
            out_offsets = n[:, None] * C * HW + c[None, :] * HW + hw
        mask = active_c[None, :]

        y = tl.load(y_ptr + cl_offsets, mask=mask, other=0.0).to(tl.float32)
        add2 = tl.load(add2_bf16_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
        out = (add2 - (y - mean1[None, :]) * correction[None, :] - mean_out[None, :]) * out_scale[None, :]
        out_bf16 = out.to(tl.bfloat16)
        tl.store(out5_ptr + out_offsets, out_bf16, mask=mask)
        sum6_acc += tl.sum(tl.where(mask, out_bf16.to(tl.float32), 0.0), axis=0)

    tile = tl.program_id(1)
    tl.store(partial6_ptr + c * HW_TILES + tile, sum6_acc, mask=c_mask)


@triton.jit
def _sum6_kernel(
    partial6_ptr,
    out6_ptr,
    C: tl.constexpr,
    HW_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (c[:, None] < C) & (tiles[None, :] < HW_TILES)
    values = tl.load(partial6_ptr + c[:, None] * HW_TILES + tiles[None, :], mask=mask, other=0.0).to(tl.float32)
    summed = tl.sum(values, axis=1).to(tl.bfloat16).to(tl.float32)
    tl.store(out6_ptr + c, summed, mask=c < C)

def _launch(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_R: int,
    BLOCK_HW: int,
    num_warps_first: int,
    num_warps_main: int,
    num_warps_final: int,
):
    x, y, mean1, inv1, weight1, bias1, spatial_bias, mean2, gamma2, beta2, residual = inputs
    _, channels, height, width = x.shape
    hw = height * width
    total_r = N * hw
    first_tiles = triton.cdiv(total_r, BLOCK_R)
    hw_tiles = triton.cdiv(hw, BLOCK_HW)
    residual_channels_last = residual.stride(1) == 1

    first_partials = torch.empty((2, channels, first_tiles), device=x.device, dtype=torch.float32)
    second_partials = torch.empty((2, channels, hw_tiles), device=x.device, dtype=torch.float32)
    stats1 = torch.empty((3, channels), device=x.device, dtype=torch.float32)
    stats2 = torch.empty((3, channels), device=x.device, dtype=torch.float32)
    partial6 = torch.empty((channels, hw_tiles), device=x.device, dtype=torch.float32)
    add2_bf16 = torch.empty_strided(tuple(residual.shape), tuple(residual.stride()), device=x.device, dtype=torch.bfloat16)

    out0 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out1 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out2 = torch.empty_strided((1, channels, height, width), (channels * hw, hw, width, 1), device=x.device, dtype=torch.float32)
    out3 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out4 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out5 = torch.empty_strided(tuple(residual.shape), tuple(residual.stride()), device=x.device, dtype=torch.bfloat16)
    out6 = torch.empty((channels,), device=x.device, dtype=torch.float32)

    c_grid = triton.cdiv(channels, BLOCK_C)
    _first_partials_kernel[(c_grid, first_tiles)](
        x,
        y,
        mean1,
        inv1,
        weight1,
        bias1,
        spatial_bias,
        mean2,
        first_partials,
        C=channels,
        H=height,
        W=width,
        HW=hw,
        TOTAL_R=total_r,
        NUM_TILES=first_tiles,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps_first,
        num_stages=4,
    )
    _stats1_kernel[(c_grid,)](
        first_partials,
        gamma2,
        beta2,
        out0,
        out1,
        stats1,
        C=channels,
        NUM_TILES=first_tiles,
        INV_REDUCE_VALUE=INV_REDUCE,
        BLOCK_C=BLOCK_C,
        BLOCK_TILES=triton.next_power_of_2(first_tiles),
        num_warps=num_warps_final,
        num_stages=1,
    )
    _phase2_kernel[(c_grid, hw_tiles)](
        x,
        y,
        mean1,
        inv1,
        weight1,
        bias1,
        spatial_bias,
        mean2,
        residual,
        stats1,
        out2,
        add2_bf16,
        second_partials,
        C=channels,
        H=height,
        W=width,
        HW=hw,
        HW_TILES=hw_tiles,
        RESIDUAL_CHANNELS_LAST=residual_channels_last,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps_main,
        num_stages=4,
    )
    _stats2_kernel[(c_grid,)](
        second_partials,
        inv1,
        weight1,
        out3,
        out4,
        stats2,
        C=channels,
        HW_TILES=hw_tiles,
        INV_REDUCE_VALUE=INV_REDUCE,
        BLOCK_C=BLOCK_C,
        BLOCK_TILES=triton.next_power_of_2(hw_tiles),
        num_warps=num_warps_final,
        num_stages=1,
    )
    _phase3_kernel[(c_grid, hw_tiles)](
        y,
        mean1,
        add2_bf16,
        stats2,
        out5,
        partial6,
        C=channels,
        H=height,
        W=width,
        HW=hw,
        HW_TILES=hw_tiles,
        OUTPUT_CHANNELS_LAST=residual_channels_last,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps_main,
        num_stages=4,
    )
    _sum6_kernel[(c_grid,)](
        partial6,
        out6,
        C=channels,
        HW_TILES=hw_tiles,
        BLOCK_C=BLOCK_C,
        BLOCK_TILES=triton.next_power_of_2(hw_tiles),
        num_warps=num_warps_final,
        num_stages=1,
    )
    return out0, out1, out2, out3, out4, out5, out6


@oracle_impl(hardware="B200", point="ff30dd34", BLOCK_C=4, BLOCK_R=1024, BLOCK_HW=8, num_warps_first=8, num_warps_main=4, num_warps_final=1)
@oracle_impl(hardware="B200", point="7a6295cd", BLOCK_C=4, BLOCK_R=1024, BLOCK_HW=7, num_warps_first=8, num_warps_main=4, num_warps_final=1)
@oracle_impl(hardware="B200", point="fd9590cc", BLOCK_C=4, BLOCK_R=1024, BLOCK_HW=7, num_warps_first=8, num_warps_main=4, num_warps_final=1)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_R: int, BLOCK_HW: int, num_warps_first: int, num_warps_main: int, num_warps_final: int):
    return _launch(
        inputs,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        BLOCK_HW=BLOCK_HW,
        num_warps_first=num_warps_first,
        num_warps_main=num_warps_main,
        num_warps_final=num_warps_final,
    )
