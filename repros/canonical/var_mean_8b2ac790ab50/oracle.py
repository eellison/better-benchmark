"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full bf16 GhostNet training-BatchNorm scope by reducing per-channel variance and mean over the fp32-cast NHWC activation, updating both running-stat copy_ aliases, preserving the returned invstd and mean side outputs, rounding the affine BN result to bf16 before the fixed channel cat, and writing the final bf16 concat-plus-residual tensor directly, whereas Inductor schedules the var_mean, running-stat updates, broadcast affine normalization, bf16 cast, cat, and residual add as generic reduction and pointwise/cat work around intermediate tensors; Inductor cannot do this today because its normalization scheduler does not preserve mutable BN side outputs and explicit bf16 cast boundaries while sinking the following static channel concat and residual add into the normalization epilogue; the fix is SCHEDULER_FUSION: teach the BN-training template to expose mean/invstd/running-stat epilogues and fuse static channel-cat plus residual-add stores into the same full-scope schedule."""

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
    elements_per_channel: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    c_mask = c_offsets < channels
    r_mask = r_offsets < elements_per_channel
    offsets = r_offsets[None, :] * channels + c_offsets[:, None]
    mask = r_mask[None, :] & c_mask[:, None]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(x, axis=1)
    sums2 = tl.sum(_f32_mul(x, x), axis=1)
    out_offsets = tl.program_id(1) * channels + c_offsets
    tl.store(partial_ptr + out_offsets, sums, mask=c_mask)
    tl.store(partial_ptr + num_chunks * channels + out_offsets, sums2, mask=c_mask)


@triton.jit
def _finalize_stats_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
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
    var = _f32_sub(total2 / elements_per_channel, _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, eps))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(
        _f32_mul(mean, momentum),
        _f32_mul(old_mean, 1.0 - momentum),
    )
    corrected_var = _f32_mul(var, running_var_correction)
    new_var = _f32_add(
        _f32_mul(corrected_var, momentum),
        _f32_mul(old_var, 1.0 - momentum),
    )

    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)


@triton.jit
def _cat_residual_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    residual_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    channels: tl.constexpr,
    out_channels: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < total_elements
    channel_out = offsets - (offsets // out_channels) * out_channels
    spatial = offsets // out_channels
    first_half = channel_out < channels
    channel_skip = tl.where(first_half, channel_out, 0)
    channel_bn = tl.where(first_half, 0, channel_out - channels)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    skip = tl.load(
        skip_ptr + spatial * channels + channel_skip,
        mask=mask & first_half,
        other=0.0,
    ).to(tl.float32)

    x = tl.load(
        x_ptr + spatial * channels + channel_bn,
        mask=mask & ~first_half,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)

    norm = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(norm, weight), bias)
    rounded_affine = affine.to(tl.bfloat16).to(tl.float32)

    cat_value = tl.where(first_half, skip, rounded_affine)
    out = _f32_add(cat_value, residual)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# 26c28da3: (T([512,80,7,7], bf16), T([80], f32), T([80], f32), T([80], f32), T([80], f32), T([512,80,7,7], bf16), T([512,160,7,7], bf16))
@oracle_impl(hardware="B200", point="26c28da3", BLOCK_R=1024, BLOCK_C=16, BLOCK_E=512, num_warps_stats=4, num_warps_final=1, num_warps_out=4)
# 1f6ad890: (T([512,56,14,14], bf16), T([56], f32), T([56], f32), T([56], f32), T([56], f32), T([512,56,14,14], bf16), T([512,112,14,14], bf16))
@oracle_impl(hardware="B200", point="1f6ad890", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_out=4)
# 4585624e: (T([512,40,14,14], bf16), T([40], f32), T([40], f32), T([40], f32), T([40], f32), T([512,40,14,14], bf16), T([512,80,14,14], bf16))
@oracle_impl(hardware="B200", point="4585624e", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_out=4)
# 25930a87: (T([512,20,28,28], bf16), T([20], f32), T([20], f32), T([20], f32), T([20], f32), T([512,20,28,28], bf16), T([512,40,28,28], bf16))
@oracle_impl(hardware="B200", point="25930a87", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_out=4)
# c94413ac: (T([512,12,56,56], bf16), T([12], f32), T([12], f32), T([12], f32), T([12], f32), T([512,12,56,56], bf16), T([512,24,56,56], bf16))
@oracle_impl(hardware="B200", point="c94413ac", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_out=4)
# 23941270: (T([512,8,112,112], bf16), T([8], f32), T([8], f32), T([8], f32), T([8], f32), T([512,8,112,112], bf16), T([512,16,112,112], bf16))
@oracle_impl(hardware="B200", point="23941270", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024, num_warps_stats=8, num_warps_final=1, num_warps_out=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_R,
    BLOCK_C,
    BLOCK_E,
    num_warps_stats,
    num_warps_final,
    num_warps_out,
):
    x, running_mean, running_var, weight, bias, skip, residual = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    out_channels = channels * 2
    hw_size = height * width
    elements_per_channel = batch * hw_size
    output_elements = batch * out_channels * hw_size
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    mean = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (channels,),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )
    partial = torch.empty_strided(
        (2, num_chunks, channels),
        (num_chunks * channels, channels, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        tuple(residual.shape),
        tuple(residual.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _partial_stats_kernel[(triton.cdiv(channels, BLOCK_C), num_chunks)](
        x,
        partial,
        channels=channels,
        elements_per_channel=elements_per_channel,
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
    _cat_residual_kernel[(triton.cdiv(output_elements, BLOCK_E),)](
        x,
        weight,
        bias,
        skip,
        residual,
        mean,
        invstd,
        out,
        total_elements=output_elements,
        channels=channels,
        out_channels=out_channels,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_out,
        num_stages=3,
    )
    return invstd, out, mean, running_mean, running_var
