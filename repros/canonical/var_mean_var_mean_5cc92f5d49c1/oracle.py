"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 dual training-BatchNorm scope for the RepVGG/ResNet block, including both population var_mean reductions over N,H,W, both saved invstd and mean outputs, the four mutable running-stat copy_ aliases, the two affine bf16 cast boundaries, the bf16 branch add, and the final ReLU in one full-scope registered implementation, whereas Inductor lowers the sibling BN-training reductions and downstream add/ReLU through generic normalization and pointwise schedules; Inductor cannot do this today because its normalization scheduler does not co-schedule paired mutable BN reductions with shared post-add consumers while preserving bf16 rounding boundaries, aliases, and channels-last/NCHW strides across all captured points; the fix is SCHEDULER_FUSION: add a guarded dual-training-BN lowering that groups sibling channel statistics, running-stat updates, saved-stat returns, and the rounded activation epilogue in one benchmark-gated schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dual_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    partial_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    x0_stride_n: tl.constexpr,
    x0_stride_c: tl.constexpr,
    x0_stride_h: tl.constexpr,
    x0_stride_w: tl.constexpr,
    x1_stride_n: tl.constexpr,
    x1_stride_c: tl.constexpr,
    x1_stride_h: tl.constexpr,
    x1_stride_w: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunk = tl.program_id(1)
    r_offsets = chunk * BLOCK_R + tl.arange(0, BLOCK_R)

    r_mask = r_offsets < elements_per_channel
    c_mask = c_offsets < channels
    n = r_offsets // hw_size
    hw = r_offsets - n * hw_size
    h = hw // width
    w = hw - h * width
    x0_offsets = (
        n[None, :] * x0_stride_n
        + c_offsets[:, None] * x0_stride_c
        + h[None, :] * x0_stride_h
        + w[None, :] * x0_stride_w
    )
    x1_offsets = (
        n[None, :] * x1_stride_n
        + c_offsets[:, None] * x1_stride_c
        + h[None, :] * x1_stride_h
        + w[None, :] * x1_stride_w
    )
    mask = c_mask[:, None] & r_mask[None, :]

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)

    mean_acc0 = tl.zeros([BLOCK_C, BLOCK_R], tl.float32)
    m2_acc0 = tl.zeros([BLOCK_C, BLOCK_R], tl.float32)
    weight_acc0 = tl.zeros([BLOCK_C, BLOCK_R], tl.float32)
    mean_next0, m2_next0, weight_next0 = triton_helpers.welford_reduce(
        x0,
        mean_acc0,
        m2_acc0,
        weight_acc0,
        True,
    )
    mean_acc0 = tl.where(mask, mean_next0, mean_acc0)
    m2_acc0 = tl.where(mask, m2_next0, m2_acc0)
    weight_acc0 = tl.where(mask, weight_next0, weight_acc0)
    block_mean0, block_m20, block_weight0 = triton_helpers.welford(
        mean_acc0,
        m2_acc0,
        weight_acc0,
        1,
    )

    mean_acc1 = tl.zeros([BLOCK_C, BLOCK_R], tl.float32)
    m2_acc1 = tl.zeros([BLOCK_C, BLOCK_R], tl.float32)
    weight_acc1 = tl.zeros([BLOCK_C, BLOCK_R], tl.float32)
    mean_next1, m2_next1, weight_next1 = triton_helpers.welford_reduce(
        x1,
        mean_acc1,
        m2_acc1,
        weight_acc1,
        True,
    )
    mean_acc1 = tl.where(mask, mean_next1, mean_acc1)
    m2_acc1 = tl.where(mask, m2_next1, m2_acc1)
    weight_acc1 = tl.where(mask, weight_next1, weight_acc1)
    block_mean1, block_m21, block_weight1 = triton_helpers.welford(
        mean_acc1,
        m2_acc1,
        weight_acc1,
        1,
    )

    out_offsets = chunk * channels + c_offsets
    plane_stride = num_chunks * channels
    tl.store(partial_ptr + out_offsets, block_mean0, mask=c_mask)
    tl.store(partial_ptr + plane_stride + out_offsets, block_m20, mask=c_mask)
    tl.store(partial_ptr + 2 * plane_stride + out_offsets, block_weight0, mask=c_mask)
    tl.store(partial_ptr + 3 * plane_stride + out_offsets, block_mean1, mask=c_mask)
    tl.store(partial_ptr + 4 * plane_stride + out_offsets, block_m21, mask=c_mask)
    tl.store(partial_ptr + 5 * plane_stride + out_offsets, block_weight1, mask=c_mask)


@triton.jit
def _dual_finalize_stats_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    invstd0_ptr,
    invstd1_ptr,
    mean0_ptr,
    mean1_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_chunks: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (c_offsets[:, None] < channels) & (chunks[None, :] < num_chunks)
    offsets = chunks[None, :] * channels + c_offsets[:, None]
    plane_stride = num_chunks * channels

    mean_inputs0 = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m2_inputs0 = tl.load(
        partial_ptr + plane_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_inputs0 = tl.load(
        partial_ptr + 2 * plane_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean_inputs1 = tl.load(
        partial_ptr + 3 * plane_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    m2_inputs1 = tl.load(
        partial_ptr + 4 * plane_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_inputs1 = tl.load(
        partial_ptr + 5 * plane_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mean_inputs0 = tl.where(mask, mean_inputs0, 0.0)
    m2_inputs0 = tl.where(mask, m2_inputs0, 0.0)
    weight_inputs0 = tl.where(mask, weight_inputs0, 0.0)
    mean_inputs1 = tl.where(mask, mean_inputs1, 0.0)
    m2_inputs1 = tl.where(mask, m2_inputs1, 0.0)
    weight_inputs1 = tl.where(mask, weight_inputs1, 0.0)

    mean0, m20, _weight0 = triton_helpers.welford(
        mean_inputs0,
        m2_inputs0,
        weight_inputs0,
        1,
    )
    mean1, m21, _weight1 = triton_helpers.welford(
        mean_inputs1,
        m2_inputs1,
        weight_inputs1,
        1,
    )
    var0 = m20 / elements_per_channel
    var1 = m21 / elements_per_channel
    invstd0 = libdevice.rsqrt(_f32_add(var0, eps))
    invstd1 = libdevice.rsqrt(_f32_add(var1, eps))

    channel_mask = c_offsets < channels
    old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    keep = 1.0 - momentum
    new_mean0 = _f32_add(_f32_mul(old_mean0, keep), _f32_mul(mean0, momentum))
    new_var0 = _f32_add(
        _f32_mul(old_var0, keep),
        _f32_mul(_f32_mul(var0, running_var_correction), momentum),
    )
    new_mean1 = _f32_add(_f32_mul(old_mean1, keep), _f32_mul(mean1, momentum))
    new_var1 = _f32_add(
        _f32_mul(old_var1, keep),
        _f32_mul(_f32_mul(var1, running_var_correction), momentum),
    )

    tl.store(invstd0_ptr + c_offsets, invstd0, mask=channel_mask)
    tl.store(invstd1_ptr + c_offsets, invstd1, mask=channel_mask)
    tl.store(mean0_ptr + c_offsets, mean0, mask=channel_mask)
    tl.store(mean1_ptr + c_offsets, mean1, mask=channel_mask)
    tl.store(running_mean0_ptr + c_offsets, new_mean0, mask=channel_mask)
    tl.store(running_var0_ptr + c_offsets, new_var0, mask=channel_mask)
    tl.store(running_mean1_ptr + c_offsets, new_mean1, mask=channel_mask)
    tl.store(running_var1_ptr + c_offsets, new_var1, mask=channel_mask)


@triton.jit
def _dual_partial_sums_kernel(
    x0_ptr,
    x1_ptr,
    partial_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    x0_stride_n: tl.constexpr,
    x0_stride_c: tl.constexpr,
    x0_stride_h: tl.constexpr,
    x0_stride_w: tl.constexpr,
    x1_stride_n: tl.constexpr,
    x1_stride_c: tl.constexpr,
    x1_stride_h: tl.constexpr,
    x1_stride_w: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunk = tl.program_id(1)
    r_offsets = chunk * BLOCK_R + tl.arange(0, BLOCK_R)

    r_mask = r_offsets < elements_per_channel
    c_mask = c_offsets < channels
    n = r_offsets // hw_size
    hw = r_offsets - n * hw_size
    h = hw // width
    w = hw - h * width
    x0_offsets = (
        n[None, :] * x0_stride_n
        + c_offsets[:, None] * x0_stride_c
        + h[None, :] * x0_stride_h
        + w[None, :] * x0_stride_w
    )
    x1_offsets = (
        n[None, :] * x1_stride_n
        + c_offsets[:, None] * x1_stride_c
        + h[None, :] * x1_stride_h
        + w[None, :] * x1_stride_w
    )
    mask = c_mask[:, None] & r_mask[None, :]

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    sum0 = tl.sum(x0, axis=1)
    sum1 = tl.sum(x1, axis=1)
    sumsq0 = tl.sum(_f32_mul(x0, x0), axis=1)
    sumsq1 = tl.sum(_f32_mul(x1, x1), axis=1)

    out_offsets = chunk * channels + c_offsets
    plane_stride = num_chunks * channels
    tl.store(partial_ptr + out_offsets, sum0, mask=c_mask)
    tl.store(partial_ptr + plane_stride + out_offsets, sumsq0, mask=c_mask)
    tl.store(partial_ptr + 2 * plane_stride + out_offsets, sum1, mask=c_mask)
    tl.store(partial_ptr + 3 * plane_stride + out_offsets, sumsq1, mask=c_mask)


@triton.jit
def _dual_finalize_sums_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    invstd0_ptr,
    invstd1_ptr,
    mean0_ptr,
    mean1_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_chunks: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (c_offsets[:, None] < channels) & (chunks[None, :] < num_chunks)
    offsets = chunks[None, :] * channels + c_offsets[:, None]
    plane_stride = num_chunks * channels

    sum0 = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=1)
    sumsq0 = tl.sum(
        tl.load(partial_ptr + plane_stride + offsets, mask=mask, other=0.0),
        axis=1,
    )
    sum1 = tl.sum(
        tl.load(partial_ptr + 2 * plane_stride + offsets, mask=mask, other=0.0),
        axis=1,
    )
    sumsq1 = tl.sum(
        tl.load(partial_ptr + 3 * plane_stride + offsets, mask=mask, other=0.0),
        axis=1,
    )

    inv_count = 1.0 / elements_per_channel
    mean0 = _f32_mul(sum0, inv_count)
    mean1 = _f32_mul(sum1, inv_count)
    var0 = _f32_sub(_f32_mul(sumsq0, inv_count), _f32_mul(mean0, mean0))
    var1 = _f32_sub(_f32_mul(sumsq1, inv_count), _f32_mul(mean1, mean1))
    var0 = tl.where(var0 < 0.0, 0.0, var0)
    var1 = tl.where(var1 < 0.0, 0.0, var1)
    invstd0 = libdevice.rsqrt(_f32_add(var0, eps))
    invstd1 = libdevice.rsqrt(_f32_add(var1, eps))

    channel_mask = c_offsets < channels
    old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + c_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    keep = 1.0 - momentum
    new_mean0 = _f32_add(_f32_mul(old_mean0, keep), _f32_mul(mean0, momentum))
    new_var0 = _f32_add(
        _f32_mul(old_var0, keep),
        _f32_mul(_f32_mul(var0, running_var_correction), momentum),
    )
    new_mean1 = _f32_add(_f32_mul(old_mean1, keep), _f32_mul(mean1, momentum))
    new_var1 = _f32_add(
        _f32_mul(old_var1, keep),
        _f32_mul(_f32_mul(var1, running_var_correction), momentum),
    )

    tl.store(invstd0_ptr + c_offsets, invstd0, mask=channel_mask)
    tl.store(invstd1_ptr + c_offsets, invstd1, mask=channel_mask)
    tl.store(mean0_ptr + c_offsets, mean0, mask=channel_mask)
    tl.store(mean1_ptr + c_offsets, mean1, mask=channel_mask)
    tl.store(running_mean0_ptr + c_offsets, new_mean0, mask=channel_mask)
    tl.store(running_var0_ptr + c_offsets, new_var0, mask=channel_mask)
    tl.store(running_mean1_ptr + c_offsets, new_mean1, mask=channel_mask)
    tl.store(running_var1_ptr + c_offsets, new_var1, mask=channel_mask)


@triton.jit
def _dual_affine_relu_kernel(
    x0_ptr,
    x1_ptr,
    weight0_ptr,
    bias0_ptr,
    weight1_ptr,
    bias1_ptr,
    invstd0_ptr,
    invstd1_ptr,
    mean0_ptr,
    mean1_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    is_channels_last: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < total_elements

    if is_channels_last:
        channel = offsets - (offsets // channels) * channels
    else:
        channel = (offsets // hw_size) - ((offsets // hw_size) // channels) * channels

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd0 = tl.load(invstd0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd1 = tl.load(invstd1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    y0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0), invstd0), weight0), bias0)
    y1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1), invstd1), weight1), bias1)
    rounded0 = y0.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    rounded1 = y1.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    summed = _f32_add(rounded0, rounded1).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    ).to(tl.float32)
    relu = tl.where(summed != summed, summed, tl.maximum(summed, 0.0))
    tl.store(
        out_ptr + offsets,
        relu.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="88ea0b10", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="df3706c4", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="44eff697", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="7e4f9b68", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="45788330", BLOCK_R=1024, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=4, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="0fb91edf", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="5550fafe", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="aa800003", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="28724d08", BLOCK_R=1024, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=4, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="3d20b12d", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="772f143b", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="2e744529", BLOCK_R=1024, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=4, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="a1b7b33d", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="a531ff95", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="ec58f56f", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="86dffc5f", BLOCK_R=512, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=4, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="e69c320b", BLOCK_R=1024, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=4, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="d120b09e", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="638131b9", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="8d9f27d9", BLOCK_R=1024, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=4, num_warps_final=1, num_warps_act=4, USE_WELFORD=True)
@oracle_impl(hardware="B200", point="f649f89b", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="e3da94b7", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="508a47f0", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8, USE_WELFORD=True)
def oracle_forward(
    inputs,
    *,
    BLOCK_R,
    BLOCK_C,
    BLOCK_E,
    num_warps_stats,
    num_warps_final,
    num_warps_act,
    USE_WELFORD=False,
):
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
    ) = inputs
    batch = int(x0.shape[0])
    channels = int(x0.shape[1])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    hw_size = height * width
    elements_per_channel = batch * hw_size
    total_elements = batch * channels * hw_size
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    invstd0 = torch.empty_strided((channels,), (1,), device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty_strided((channels,), (1,), device=x0.device, dtype=torch.float32)
    mean0 = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    mean1 = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    partial_planes = 6 if USE_WELFORD else 4
    partial = torch.empty_strided(
        (partial_planes, num_chunks, channels),
        (num_chunks * channels, channels, 1),
        device=x0.device,
        dtype=torch.float32,
    )

    if USE_WELFORD:
        _dual_partial_stats_kernel[(triton.cdiv(channels, BLOCK_C), num_chunks)](
            x0,
            x1,
            partial,
            channels=channels,
            width=width,
            hw_size=hw_size,
            elements_per_channel=elements_per_channel,
            x0_stride_n=int(x0.stride(0)),
            x0_stride_c=int(x0.stride(1)),
            x0_stride_h=int(x0.stride(2)),
            x0_stride_w=int(x0.stride(3)),
            x1_stride_n=int(x1.stride(0)),
            x1_stride_c=int(x1.stride(1)),
            x1_stride_h=int(x1.stride(2)),
            x1_stride_w=int(x1.stride(3)),
            num_chunks=num_chunks,
            BLOCK_C=BLOCK_C,
            BLOCK_R=BLOCK_R,
            num_warps=num_warps_stats,
            num_stages=3,
        )
        _dual_finalize_stats_kernel[(triton.cdiv(channels, BLOCK_C),)](
            partial,
            running_mean0,
            running_var0,
            running_mean1,
            running_var1,
            invstd0,
            invstd1,
            mean0,
            mean1,
            channels=channels,
            elements_per_channel=elements_per_channel,
            num_chunks=num_chunks,
            eps=EPS,
            momentum=MOMENTUM,
            running_var_correction=RUNNING_VAR_CORRECTION,
            BLOCK_CHUNKS=block_chunks,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps_final,
            num_stages=3,
        )
    else:
        _dual_partial_sums_kernel[(triton.cdiv(channels, BLOCK_C), num_chunks)](
            x0,
            x1,
            partial,
            channels=channels,
            width=width,
            hw_size=hw_size,
            elements_per_channel=elements_per_channel,
            x0_stride_n=int(x0.stride(0)),
            x0_stride_c=int(x0.stride(1)),
            x0_stride_h=int(x0.stride(2)),
            x0_stride_w=int(x0.stride(3)),
            x1_stride_n=int(x1.stride(0)),
            x1_stride_c=int(x1.stride(1)),
            x1_stride_h=int(x1.stride(2)),
            x1_stride_w=int(x1.stride(3)),
            num_chunks=num_chunks,
            BLOCK_C=BLOCK_C,
            BLOCK_R=BLOCK_R,
            num_warps=num_warps_stats,
            num_stages=3,
        )
        _dual_finalize_sums_kernel[(triton.cdiv(channels, BLOCK_C),)](
            partial,
            running_mean0,
            running_var0,
            running_mean1,
            running_var1,
            invstd0,
            invstd1,
            mean0,
            mean1,
            channels=channels,
            elements_per_channel=elements_per_channel,
            num_chunks=num_chunks,
            eps=EPS,
            momentum=MOMENTUM,
            running_var_correction=RUNNING_VAR_CORRECTION,
            BLOCK_CHUNKS=block_chunks,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps_final,
            num_stages=3,
        )
    _dual_affine_relu_kernel[(triton.cdiv(total_elements, BLOCK_E),)](
        x0,
        x1,
        weight0,
        bias0,
        weight1,
        bias1,
        invstd0,
        invstd1,
        mean0,
        mean1,
        out,
        total_elements=total_elements,
        channels=channels,
        hw_size=hw_size,
        is_channels_last=int(x0.stride(1)) == 1,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_act,
        num_stages=3,
    )
    return (
        invstd0,
        invstd1,
        out,
        mean1,
        mean0,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )
