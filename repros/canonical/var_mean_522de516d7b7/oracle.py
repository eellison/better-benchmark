"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileNetV3 training-BatchNorm scope with split channel statistics, returned mean and rsqrt tensors, mutable running-stat copy_ side effects, the affine bf16 rounding boundary, and the final hard-swish bf16 activation in one registered full-scope implementation, whereas Inductor lowers the captured var_mean/update/normalization/cast/clamp graph through generic reduction and pointwise schedules; Inductor cannot do this today because its normalization scheduler does not expose a reusable training-BN template that keeps copy_ side effects, saved statistics, explicit bf16 cast boundaries, and the activation epilogue in one full graph plan across both channels-last and contiguous NCHW points; the fix is SCHEDULER_FUSION: add a guarded BN-training hard-swish lowering that emits split channel stats, running-stat updates, saved-stat outputs, and the rounded activation epilogue as one benchmark-gated schedule."""

import torch
import triton
import triton.language as tl
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
def _partial_stats_kernel(
    x_ptr,
    partial_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
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
    x_offsets = (
        n[None, :] * x_stride_n
        + c_offsets[:, None] * x_stride_c
        + h[None, :] * x_stride_h
        + w[None, :] * x_stride_w
    )
    mask = c_mask[:, None] & r_mask[None, :]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(x, axis=1)
    sums2 = tl.sum(_f32_mul(x, x), axis=1)
    out_offsets = chunk * channels + c_offsets
    tl.store(partial_ptr + out_offsets, sums, mask=c_mask)
    tl.store(partial_ptr + num_chunks * channels + out_offsets, sums2, mask=c_mask)


@triton.jit
def _finalize_stats_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_out_ptr,
    rsqrt_out_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_chunks: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    channel = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < num_chunks
    offsets = chunks * channels + channel

    sums = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums2 = tl.load(
        partial_ptr + num_chunks * channels + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(sums, axis=0)
    total2 = tl.sum(sums2, axis=0)

    mean = total / elements_per_channel
    mean_sq = _f32_mul(mean, mean)
    var = _f32_sub(total2 / elements_per_channel, mean_sq)
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, eps))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(old_mean, 1.0 - momentum), _f32_mul(mean, momentum))
    corrected_var = _f32_mul(var, running_var_correction)
    new_var = _f32_add(
        _f32_mul(old_var, 1.0 - momentum),
        _f32_mul(corrected_var, momentum),
    )

    tl.store(mean_out_ptr + channel, mean)
    tl.store(rsqrt_out_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)


@triton.jit
def _activation_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    norm = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(norm, weight), bias)
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    relu6 = _f32_add(rounded, 3.0)
    relu6 = tl.where(relu6 < 0.0, 0.0, relu6)
    relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
    hswish = _f32_mul(_f32_mul(rounded, relu6), 0.16666666666666666)
    tl.store(out_ptr + offsets, hswish.to(tl.bfloat16), mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="d6a317bc", BLOCK_R=4096, BLOCK_C=16, BLOCK_E=2048, num_warps_stats=8, num_warps_final=1, num_warps_act=8)
@oracle_impl(hardware="B200", point="055f7aad", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="bb4cfa64", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="c0a7cc9a", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="03d2a306", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="df1d49cf", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="66871354", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="a3c80480", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="e8a9d13a", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="8aafc432", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="ba044ce1", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="531cd9bd", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="76491bca", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="4e453d00", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="85aaacf2", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
@oracle_impl(hardware="B200", point="5e2cd32b", BLOCK_R=2048, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_act=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_R,
    BLOCK_C,
    BLOCK_E,
    num_warps_stats,
    num_warps_final,
    num_warps_act,
):
    x, running_mean, running_var, weight, bias = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw_size = height * width
    elements_per_channel = batch * hw_size
    total_elements = batch * channels * hw_size
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    mean = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )

    partial = torch.empty_strided(
        (2, num_chunks, channels),
        (num_chunks * channels, channels, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _partial_stats_kernel[(triton.cdiv(channels, BLOCK_C), num_chunks)](
        x,
        partial,
        channels=channels,
        width=width,
        hw_size=hw_size,
        elements_per_channel=elements_per_channel,
        x_stride_n=int(x.stride(0)),
        x_stride_c=int(x.stride(1)),
        x_stride_h=int(x.stride(2)),
        x_stride_w=int(x.stride(3)),
        num_chunks=num_chunks,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _finalize_stats_kernel[(channels,)](
        partial,
        running_mean,
        running_var,
        mean,
        invstd,
        channels=channels,
        elements_per_channel=elements_per_channel,
        num_chunks=num_chunks,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_CHUNKS=block_chunks,
        num_warps=num_warps_final,
        num_stages=3,
    )
    _activation_kernel[(triton.cdiv(total_elements, BLOCK_E),)](
        x,
        mean,
        invstd,
        weight,
        bias,
        out,
        total_elements=total_elements,
        channels=channels,
        hw_size=hw_size,
        is_channels_last=int(x.stride(1)) == 1,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_act,
        num_stages=3,
    )
    return mean, invstd, out, running_mean, running_var
