"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete VisFormer double-normalization backward scope, including the first channel reductions, the materialized batch-sum side output, the bf16-rounded full tensor, and the final reduction of that bf16 tensor, by reusing compact per-channel/per-spatial summaries across the dependent formulas, whereas Inductor schedules the decomposed sibling and dependent reductions as generic regions over large intermediates; Inductor cannot do this today because its algebraic simplifier and reduction scheduler do not recognize this BN-backward-style chain as reusable summaries once the rounded full-tensor and side reductions are also live; the fix is ALGEBRAIC_ELIMINATION: add a guarded multi-output reduction rewrite that preserves shared channel summaries, lowers the side-output reductions directly, and materializes the rounded epilogue only at the required cast boundary."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
INV_REDUCE = 9.964923469387754e-06


@triton.jit
def _first_batch_summary_kernel(
    x_ptr,
    y_ptr,
    mean1_ptr,
    inv1_ptr,
    weight1_ptr,
    bias1_ptr,
    spatial_bias_ptr,
    mean2_ptr,
    batch_summaries_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    HW_TILES: tl.constexpr,
    CHUNK_HW: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    c = tl.program_id(0)
    tile_id = tl.program_id(1)
    n = tile_id // HW_TILES
    hw_tile = tile_id - n * HW_TILES
    r = tl.arange(0, BLOCK_CHUNK)
    hw = hw_tile * CHUNK_HW + r
    mask = (r < CHUNK_HW) & (hw < HW)
    h = hw // W
    w = hw - h * W
    offsets = n * C * HW + h * W * C + w * C + c

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c).to(tl.float32)
    inv1 = tl.load(inv1_ptr + c).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c).to(tl.float32)
    spatial_bias = tl.load(spatial_bias_ptr + h * W * C + w * C + c, mask=mask, other=0.0).to(tl.float32)

    affine = ((y - mean1) * inv1) * weight1 + bias1
    norm = affine.to(tl.bfloat16).to(tl.float32) + spatial_bias - mean2
    total_tiles = 128 * HW_TILES
    base = c * total_tiles + tile_id
    plane = C * total_tiles
    tl.store(batch_summaries_ptr + base, tl.sum(tl.where(mask, x, 0.0), axis=0))
    tl.store(batch_summaries_ptr + plane + base, tl.sum(tl.where(mask, x * norm, 0.0), axis=0))


@triton.jit
def _stats1_from_batch_kernel(
    batch_summaries_ptr,
    gamma2_ptr,
    beta2_ptr,
    out0_ptr,
    out1_ptr,
    stats1_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    INV_REDUCE_VALUE: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    base = c * NUM_TILES + tiles
    plane = C * NUM_TILES
    sum_x_parts = tl.load(batch_summaries_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum_x_norm_parts = tl.load(batch_summaries_ptr + plane + base, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(sum_x_parts, axis=0)
    sum_x_norm = tl.sum(sum_x_norm_parts, axis=0)
    gamma2 = tl.load(gamma2_ptr + c).to(tl.float32)
    beta2 = tl.load(beta2_ptr + c).to(tl.float32)
    mean_x = sum_x * INV_REDUCE_VALUE
    norm_scale = (sum_x_norm * INV_REDUCE_VALUE) * (gamma2 * gamma2)
    grad_scale = gamma2 * beta2
    tl.store(out0_ptr + c, sum_x)
    tl.store(out1_ptr + c, sum_x_norm * gamma2)
    tl.store(stats1_ptr + c, mean_x)
    tl.store(stats1_ptr + C + c, norm_scale)
    tl.store(stats1_ptr + 2 * C + c, grad_scale)

@triton.jit
def _first_group_summary_kernel(
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
    GROUP_N: tl.constexpr,
    GROUP_REDUCE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    group = tl.program_id(1)
    r_offsets = tl.arange(0, BLOCK_R)
    sum_x_acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    sum_x_norm_acc = tl.zeros([BLOCK_R], dtype=tl.float32)

    mean1 = tl.load(mean1_ptr + c).to(tl.float32)
    inv1 = tl.load(inv1_ptr + c).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c).to(tl.float32)

    for base_r in tl.range(0, GROUP_REDUCE, BLOCK_R):
        r = base_r + r_offsets
        mask = r < GROUP_REDUCE
        n = group * GROUP_N + r // HW
        hw = r - (r // HW) * HW
        h = hw // W
        w = hw - h * W
        offsets = n * C * HW + h * W * C + w * C + c

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        spatial_bias = tl.load(spatial_bias_ptr + h * W * C + w * C + c, mask=mask, other=0.0).to(tl.float32)
        affine = ((y - mean1) * inv1) * weight1 + bias1
        norm = affine.to(tl.bfloat16).to(tl.float32) + spatial_bias - mean2
        sum_x_acc += tl.where(mask, x, 0.0)
        sum_x_norm_acc += tl.where(mask, x * norm, 0.0)

    base = c * GROUPS + group
    plane = C * GROUPS
    tl.store(partials_ptr + base, tl.sum(sum_x_acc, axis=0))
    tl.store(partials_ptr + plane + base, tl.sum(sum_x_norm_acc, axis=0))


@triton.jit
def _stats1_from_group_summary_kernel(
    partials_ptr,
    gamma2_ptr,
    beta2_ptr,
    out0_ptr,
    out1_ptr,
    stats1_ptr,
    C: tl.constexpr,
    GROUPS: tl.constexpr,
    INV_REDUCE_VALUE: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    c = tl.program_id(0)
    groups = tl.arange(0, BLOCK_G)
    mask = groups < GROUPS
    base = c * GROUPS + groups
    plane = C * GROUPS
    sum_x_parts = tl.load(partials_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum_x_norm_parts = tl.load(partials_ptr + plane + base, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(sum_x_parts, axis=0)
    sum_x_norm = tl.sum(sum_x_norm_parts, axis=0)
    gamma2 = tl.load(gamma2_ptr + c).to(tl.float32)
    beta2 = tl.load(beta2_ptr + c).to(tl.float32)
    mean_x = sum_x * INV_REDUCE_VALUE
    norm_scale = (sum_x_norm * INV_REDUCE_VALUE) * (gamma2 * gamma2)
    grad_scale = gamma2 * beta2
    tl.store(out0_ptr + c, sum_x)
    tl.store(out1_ptr + c, sum_x_norm * gamma2)
    tl.store(stats1_ptr + c, mean_x)
    tl.store(stats1_ptr + C + c, norm_scale)
    tl.store(stats1_ptr + 2 * C + c, grad_scale)


@triton.jit
def _zero_out2_kernel(
    out2_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    tl.store(out2_ptr + offsets, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)


@triton.jit
def _rounded_add2_group_partial_kernel(
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
    second_summaries_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    RESIDUAL_CHANNELS_LAST: tl.constexpr,
    GROUP_N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    group = tl.program_id(1)
    n_rel = tl.arange(0, GROUP_N)[:, None]
    hw = tl.arange(0, BLOCK_HW)[None, :]
    n = group * GROUP_N + n_rel
    h = hw // W
    w = hw - h * W
    hw_mask = hw < HW
    mask = (n < 128) & (hw < HW)
    cl_offsets = n * C * HW + h * W * C + w * C + c
    if RESIDUAL_CHANNELS_LAST:
        out_offsets = cl_offsets
    else:
        out_offsets = n * C * HW + c * HW + hw

    x = tl.load(x_ptr + cl_offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + cl_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
    spatial_bias = tl.load(spatial_bias_ptr + h * W * C + w * C + c, mask=hw_mask, other=0.0).to(tl.float32)

    mean1 = tl.load(mean1_ptr + c).to(tl.float32)
    inv1 = tl.load(inv1_ptr + c).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c).to(tl.float32)
    mean_x = tl.load(stats1_ptr + c).to(tl.float32)
    norm_scale = tl.load(stats1_ptr + C + c).to(tl.float32)
    grad_scale = tl.load(stats1_ptr + 2 * C + c).to(tl.float32)

    affine = ((y - mean1) * inv1) * weight1 + bias1
    norm = affine.to(tl.bfloat16).to(tl.float32) + spatial_bias - mean2
    add2 = residual + (x - norm * norm_scale - mean_x) * grad_scale
    add2_bf16 = add2.to(tl.bfloat16)
    add2_f32 = add2_bf16.to(tl.float32)
    sub4 = y - mean1

    hw_vec = tl.arange(0, BLOCK_HW)
    out2_part = tl.sum(tl.where(mask, add2, 0.0), axis=0)
    tl.atomic_add(out2_ptr + c * HW + hw_vec, out2_part, sem="relaxed", mask=hw_vec < HW)
    tl.store(add2_bf16_ptr + out_offsets, add2_bf16, mask=mask)
    sum_add2_hw = tl.sum(tl.where(mask, add2_f32, 0.0), axis=0)
    sum_add2_sub4_hw = tl.sum(tl.where(mask, add2_f32 * sub4, 0.0), axis=0)
    base = c * NUM_GROUPS + group
    plane = C * NUM_GROUPS
    tl.store(second_summaries_ptr + base, tl.sum(sum_add2_hw, axis=0))
    tl.store(second_summaries_ptr + plane + base, tl.sum(sum_add2_sub4_hw, axis=0))


@triton.jit
def _second_finalize_kernel(
    second_summaries_ptr,
    inv1_ptr,
    weight1_ptr,
    out3_ptr,
    out4_ptr,
    stats2_ptr,
    C: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    INV_REDUCE_VALUE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0)
    part = tl.arange(0, BLOCK_N)
    mask = part < NUM_PARTS
    base = c * NUM_PARTS + part
    plane = C * NUM_PARTS
    sum4_parts = tl.load(second_summaries_ptr + base, mask=mask, other=0.0).to(tl.float32)
    sum5_parts = tl.load(second_summaries_ptr + plane + base, mask=mask, other=0.0).to(tl.float32)
    sum4 = tl.sum(sum4_parts, axis=0)
    sum5 = tl.sum(sum5_parts, axis=0)
    inv1 = tl.load(inv1_ptr + c).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c).to(tl.float32)

    mean_out = sum4 * INV_REDUCE_VALUE
    correction = (sum5 * INV_REDUCE_VALUE) * (inv1 * inv1)
    out_scale = inv1 * weight1

    tl.store(out3_ptr + c, sum4)
    tl.store(out4_ptr + c, sum5 * inv1)
    tl.store(stats2_ptr + c, mean_out)
    tl.store(stats2_ptr + C + c, correction)
    tl.store(stats2_ptr + 2 * C + c, out_scale)


@triton.jit
def _epilogue_partial_kernel(
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
    TOTAL_REDUCE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    OUTPUT_CHANNELS_LAST: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    r = tile * BLOCK_R + tl.arange(0, BLOCK_R)
    n = r // HW
    hw = r - n * HW
    h = hw // W
    w = hw - h * W
    mask = r < TOTAL_REDUCE
    cl_offsets = n * C * HW + h * W * C + w * C + c
    if OUTPUT_CHANNELS_LAST:
        out_offsets = cl_offsets
    else:
        out_offsets = n * C * HW + c * HW + hw

    y = tl.load(y_ptr + cl_offsets, mask=mask, other=0.0).to(tl.float32)
    add2 = tl.load(add2_bf16_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c).to(tl.float32)
    mean_out = tl.load(stats2_ptr + c).to(tl.float32)
    correction = tl.load(stats2_ptr + C + c).to(tl.float32)
    out_scale = tl.load(stats2_ptr + 2 * C + c).to(tl.float32)

    out = (add2 - (y - mean1) * correction - mean_out) * out_scale
    out_bf16 = out.to(tl.bfloat16)
    tl.store(out5_ptr + out_offsets, out_bf16, mask=mask)
    partial = tl.sum(tl.where(mask, out_bf16.to(tl.float32), 0.0), axis=0)
    tl.store(partial6_ptr + c * NUM_TILES + tile, partial)


@triton.jit
def _final_sum6_kernel(
    partial6_ptr,
    out6_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    values = tl.load(partial6_ptr + c * NUM_TILES + tiles, mask=mask, other=0.0).to(tl.float32)
    summed = tl.sum(values, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out6_ptr + c, summed)


def _launch(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_HW: int,
    CHUNK_HW: int,
    FIRST_GROUPS: int,
    FIRST_BLOCK_R: int,
    SECOND_GROUP_N: int,
    BLOCK_R: int,
    num_warps_first: int,
    num_warps_summary: int,
    num_warps_reduce: int,
):
    x, y, mean1, inv1, weight1, bias1, spatial_bias, mean2, gamma2, beta2, residual = inputs
    _, channels, height, width = x.shape
    hw = height * width
    total_reduce = N * hw
    num_tiles = triton.cdiv(total_reduce, BLOCK_R)
    second_groups = triton.cdiv(N, SECOND_GROUP_N)
    residual_channels_last = residual.stride(1) == 1
    block_hw_all = triton.next_power_of_2(hw)

    second_summaries = torch.empty((2, channels, second_groups), device=x.device, dtype=torch.float32)
    stats1 = torch.empty((3, channels), device=x.device, dtype=torch.float32)
    stats2 = torch.empty((3, channels), device=x.device, dtype=torch.float32)
    partial6 = torch.empty((channels, num_tiles), device=x.device, dtype=torch.float32)
    add2_bf16 = torch.empty_strided(tuple(residual.shape), tuple(residual.stride()), device=x.device, dtype=torch.bfloat16)

    out0 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out1 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out2 = torch.empty_strided((1, channels, height, width), (channels * hw, hw, width, 1), device=x.device, dtype=torch.float32)
    out3 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out4 = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out5 = torch.empty_strided(tuple(residual.shape), tuple(residual.stride()), device=x.device, dtype=torch.bfloat16)
    out6 = torch.empty((channels,), device=x.device, dtype=torch.float32)

    if FIRST_GROUPS == 0:
        first_hw_tiles = triton.cdiv(hw, CHUNK_HW)
        first_tiles = N * first_hw_tiles
        batch_summaries = torch.empty((2, channels, first_tiles), device=x.device, dtype=torch.float32)
        _first_batch_summary_kernel[(channels, first_tiles)](
            x,
            y,
            mean1,
            inv1,
            weight1,
            bias1,
            spatial_bias,
            mean2,
            batch_summaries,
            C=channels,
            H=height,
            W=width,
            HW=hw,
            HW_TILES=first_hw_tiles,
            CHUNK_HW=CHUNK_HW,
            BLOCK_CHUNK=triton.next_power_of_2(CHUNK_HW),
            num_warps=num_warps_summary,
            num_stages=4,
        )
        _stats1_from_batch_kernel[(channels,)](
            batch_summaries,
            gamma2,
            beta2,
            out0,
            out1,
            stats1,
            C=channels,
            NUM_TILES=first_tiles,
            INV_REDUCE_VALUE=INV_REDUCE,
            BLOCK_TILES=triton.next_power_of_2(first_tiles),
            num_warps=num_warps_summary,
            num_stages=1,
        )
    else:
        first_group_n = N // FIRST_GROUPS
        first_group_reduce = first_group_n * hw
        first_partials = torch.empty((2, channels, FIRST_GROUPS), device=x.device, dtype=torch.float32)
        _first_group_summary_kernel[(channels, FIRST_GROUPS)](
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
            GROUP_N=first_group_n,
            GROUP_REDUCE=first_group_reduce,
            GROUPS=FIRST_GROUPS,
            BLOCK_R=FIRST_BLOCK_R,
            num_warps=num_warps_first,
            num_stages=1,
        )
        _stats1_from_group_summary_kernel[(channels,)](
            first_partials,
            gamma2,
            beta2,
            out0,
            out1,
            stats1,
            C=channels,
            GROUPS=FIRST_GROUPS,
            INV_REDUCE_VALUE=INV_REDUCE,
            BLOCK_G=triton.next_power_of_2(FIRST_GROUPS),
            num_warps=num_warps_summary,
            num_stages=1,
        )
    _zero_out2_kernel[(triton.cdiv(channels * hw, 1024),)](
        out2,
        TOTAL=channels * hw,
        BLOCK=1024,
        num_warps=num_warps_summary,
        num_stages=1,
    )
    _rounded_add2_group_partial_kernel[(channels, second_groups)](
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
        second_summaries,
        C=channels,
        H=height,
        W=width,
        HW=hw,
        RESIDUAL_CHANNELS_LAST=residual_channels_last,
        GROUP_N=SECOND_GROUP_N,
        NUM_GROUPS=second_groups,
        BLOCK_HW=block_hw_all,
        num_warps=num_warps_summary,
        num_stages=4,
    )
    _second_finalize_kernel[(channels,)](
        second_summaries,
        inv1,
        weight1,
        out3,
        out4,
        stats2,
        C=channels,
        NUM_PARTS=second_groups,
        INV_REDUCE_VALUE=INV_REDUCE,
        BLOCK_C=BLOCK_C,
        BLOCK_N=triton.next_power_of_2(second_groups),
        num_warps=num_warps_summary,
        num_stages=4,
    )
    _epilogue_partial_kernel[(channels, num_tiles)](
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
        TOTAL_REDUCE=total_reduce,
        NUM_TILES=num_tiles,
        OUTPUT_CHANNELS_LAST=residual_channels_last,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps_reduce,
        num_stages=4,
    )
    _final_sum6_kernel[(channels,)](
        partial6,
        out6,
        C=channels,
        NUM_TILES=num_tiles,
        BLOCK_C=BLOCK_C,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=1,
        num_stages=1,
    )
    return out0, out1, out2, out3, out4, out5, out6


@oracle_impl(hardware="B200", point="ff30dd34", BLOCK_C=16, BLOCK_HW=4, CHUNK_HW=196, FIRST_GROUPS=0, FIRST_BLOCK_R=2048, SECOND_GROUP_N=4, BLOCK_R=1024, num_warps_first=8, num_warps_summary=2, num_warps_reduce=4)
@oracle_impl(hardware="B200", point="7a6295cd", BLOCK_C=16, BLOCK_HW=4, CHUNK_HW=196, FIRST_GROUPS=4, FIRST_BLOCK_R=2048, SECOND_GROUP_N=8, BLOCK_R=1024, num_warps_first=8, num_warps_summary=4, num_warps_reduce=4)
@oracle_impl(hardware="B200", point="fd9590cc", BLOCK_C=16, BLOCK_HW=4, CHUNK_HW=64, FIRST_GROUPS=4, FIRST_BLOCK_R=2048, SECOND_GROUP_N=8, BLOCK_R=1024, num_warps_first=8, num_warps_summary=4, num_warps_reduce=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_HW: int, CHUNK_HW: int, FIRST_GROUPS: int, FIRST_BLOCK_R: int, SECOND_GROUP_N: int, BLOCK_R: int, num_warps_first: int, num_warps_summary: int, num_warps_reduce: int):
    return _launch(
        inputs,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        CHUNK_HW=CHUNK_HW,
        FIRST_GROUPS=FIRST_GROUPS,
        FIRST_BLOCK_R=FIRST_BLOCK_R,
        SECOND_GROUP_N=SECOND_GROUP_N,
        BLOCK_R=BLOCK_R,
        num_warps_first=num_warps_first,
        num_warps_summary=num_warps_summary,
        num_warps_reduce=num_warps_reduce,
    )
