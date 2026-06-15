"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 training-BatchNorm plus skip-branch channel-shuffle scope, including fp32 population var_mean over N,H,W, eps=1e-5 rsqrt side output, running mean/variance copy_ side effects with the captured correction literal, affine epilogue with the explicit bf16 cast before ReLU, direct cat/view/permute/clone/view channel-shuffle materialization, and the full shuffled output tensor, whereas Inductor lowers the BN reduction, mutable running-stat updates, concat, permutation clone, and final view as generic reduction and layout work; Inductor cannot do this today because its scheduler does not preserve the fixed channel-shuffle consumer across a training normalization producer with observable stats, bf16 cast boundaries, side-effect outputs, and an already-strided sibling input; the fix is SCHEDULER_FUSION: teach the BN-training/layout scheduler to fuse running-stat epilogues with direct channel-shuffle stores while preserving side outputs, bf16 rounding boundaries, and the full output scope."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871


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
def _bn_partial_sums_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    BLOCK_K: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    k_block = tl.program_id(1)
    channel = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (channel[:, None] < channels) & (offsets[None, :] < elements_per_channel)
    n_idx = offsets // hw_size
    hw = offsets - n_idx * hw_size
    h_idx = hw // width
    w_idx = hw - h_idx * width

    x_offsets = (
        n_idx[None, :] * x_stride_n
        + channel[:, None] * x_stride_c
        + h_idx[None, :] * x_stride_h
        + w_idx[None, :] * x_stride_w
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float64)
    sums = tl.sum(x, axis=1)
    sums2 = tl.sum(x * x, axis=1)
    out_offsets = k_block * channels + channel
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channel < channels)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channel < channels)


@triton.jit
def _bn_finalize_sums_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channel = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channel[:, None] < channels) & (chunks[None, :] < num_chunks)
    offsets = chunks[None, :] * channels + channel[:, None]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float64)
    sums2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0).to(tl.float64)

    total = tl.sum(sums, axis=1)
    total2 = tl.sum(sums2, axis=1)
    mean64 = total * (1.0 / elements_per_channel)
    variance64 = total2 * (1.0 / elements_per_channel) - mean64 * mean64
    variance = tl.where(variance64 < 0.0, 0.0, variance64).to(tl.float32)
    mean = mean64.to(tl.float32)
    invstd = tl.rsqrt(_f32_add(variance, 1.0e-5))

    channel_mask = channel < channels
    old_mean = tl.load(running_mean_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    new_var = _f32_add(
        _f32_mul(_f32_mul(variance, 1.0001594642002871), 0.1),
        _f32_mul(old_var, 0.9),
    )

    tl.store(mean_ptr + channel, mean, mask=channel_mask)
    tl.store(invstd_ptr + channel, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channel, new_mean, mask=channel_mask)
    tl.store(running_var_ptr + channel, new_var, mask=channel_mask)


@triton.jit
def _bn_skip_shuffle_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    out_channels: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    skip_stride_n: tl.constexpr,
    skip_stride_c: tl.constexpr,
    skip_stride_h: tl.constexpr,
    skip_stride_w: tl.constexpr,
    width: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < total_elements
    hw = offsets - (offsets // hw_size) * hw_size
    channel = (offsets // hw_size) - ((offsets // hw_size) // channels) * channels
    n_idx = offsets // (channels * hw_size)
    h_idx = hw // width
    w_idx = hw - h_idx * width

    x_offsets = (
        n_idx * x_stride_n
        + channel * x_stride_c
        + h_idx * x_stride_h
        + w_idx * x_stride_w
    )
    skip_offsets = (
        n_idx * skip_stride_n
        + channel * skip_stride_c
        + h_idx * skip_stride_h
        + w_idx * skip_stride_w
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float64)
    skip = tl.load(skip_ptr + skip_offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float64)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float64)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float64)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float64)

    affine = ((x - mean) * invstd) * weight + bias
    rounded = affine.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu = tl.where((rounded > 0.0) | (rounded != rounded), rounded, 0.0)

    out_base = n_idx * out_channels * hw_size + channel * 2 * hw_size + hw
    tl.store(out_ptr + out_base, skip, mask=mask)
    tl.store(out_ptr + out_base + hw_size, relu.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# bf16[128,232,7,7] -> bf16[128,464,7,7]
@oracle_impl(hardware="B200", point="b0183d97", BLOCK_K=1024, C_BLOCK=1, BLOCK_E=1024, num_warps_stats=4, num_warps_act=4)
# bf16[128,116,14,14] -> bf16[128,232,14,14]
@oracle_impl(hardware="B200", point="00e34431", BLOCK_K=1024, C_BLOCK=1, BLOCK_E=1024, num_warps_stats=4, num_warps_act=4)
# bf16[128,58,28,28] -> bf16[128,116,28,28]
@oracle_impl(hardware="B200", point="99b3b05e", BLOCK_K=1024, C_BLOCK=1, BLOCK_E=1024, num_warps_stats=4, num_warps_act=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    C_BLOCK: int,
    BLOCK_E: int,
    num_warps_stats: int,
    num_warps_act: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw_size = height * width
    elements_per_channel = batch * hw_size
    total_elements = batch * channels * hw_size
    out_channels = channels * 2
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_K)
    block_chunks = _next_power_of_2(num_chunks)

    mean = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    shuffled = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * hw_size, hw_size, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_chunks, channels), device=arg0_1.device, dtype=torch.float64)
    partial_sum2 = torch.empty((num_chunks, channels), device=arg0_1.device, dtype=torch.float64)

    _bn_partial_sums_kernel[(triton.cdiv(channels, C_BLOCK), num_chunks)](
        arg0_1,
        partial_sum,
        partial_sum2,
        channels=channels,
        width=width,
        hw_size=hw_size,
        elements_per_channel=elements_per_channel,
        x_stride_n=int(arg0_1.stride(0)),
        x_stride_c=int(arg0_1.stride(1)),
        x_stride_h=int(arg0_1.stride(2)),
        x_stride_w=int(arg0_1.stride(3)),
        BLOCK_K=BLOCK_K,
        C_BLOCK=C_BLOCK,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _bn_finalize_sums_kernel[(triton.cdiv(channels, C_BLOCK),)](
        partial_sum,
        partial_sum2,
        arg1_1,
        arg2_1,
        mean,
        invstd,
        channels=channels,
        elements_per_channel=elements_per_channel,
        num_chunks=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=C_BLOCK,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _bn_skip_shuffle_kernel[(triton.cdiv(total_elements, BLOCK_E),)](
        arg0_1,
        arg3_1,
        arg4_1,
        arg5_1,
        mean,
        invstd,
        shuffled,
        total_elements=total_elements,
        channels=channels,
        hw_size=hw_size,
        out_channels=out_channels,
        x_stride_n=int(arg0_1.stride(0)),
        x_stride_c=int(arg0_1.stride(1)),
        x_stride_h=int(arg0_1.stride(2)),
        x_stride_w=int(arg0_1.stride(3)),
        skip_stride_n=int(arg5_1.stride(0)),
        skip_stride_c=int(arg5_1.stride(1)),
        skip_stride_h=int(arg5_1.stride(2)),
        skip_stride_w=int(arg5_1.stride(3)),
        width=width,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_act,
        num_stages=3,
    )

    return mean, invstd, shuffled, arg1_1, arg2_1
