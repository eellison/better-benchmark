"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 training-BatchNorm plus skip-branch channel-shuffle scope, including fp32 population var_mean over N,H,W, eps=1e-5 rsqrt side output, running mean/variance copy_ side effects with the captured correction literal, affine epilogue with the explicit bf16 cast before ReLU, direct cat/view/permute/clone/view channel-shuffle materialization, and returned split aliases from one backing tensor, whereas Inductor lowers the BN reduction, mutable running-stat updates, concat, permutation clone, and split as generic reduction and layout work; Inductor cannot do this today because its scheduler does not preserve the fixed channel-shuffle consumer across a training normalization producer with observable stats, bf16 cast boundaries, side-effect outputs, and an already-strided sibling input; the fix is SCHEDULER_FUSION: teach the BN-training/layout scheduler to fuse running-stat epilogues with direct channel-shuffle stores while preserving side outputs, bf16 rounding boundaries, and split-view aliases."""

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
def _bn_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < elements_per_channel
    n_idx = offsets // hw_size
    hw = offsets - n_idx * hw_size
    h_idx = hw // width
    w_idx = hw - h_idx * width

    x_offsets = (
        n_idx * x_stride_n
        + channel * x_stride_c
        + h_idx * x_stride_h
        + w_idx * x_stride_w
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float64)

    mean64 = tl.sum(x, axis=0) * (1.0 / elements_per_channel)
    centered64 = tl.where(mask, x - mean64, 0.0)
    variance64 = tl.sum(centered64 * centered64, axis=0) * (1.0 / elements_per_channel)
    variance = tl.where(variance64 < 0.0, 0.0, variance64).to(tl.float32)
    mean = mean64.to(tl.float32)
    invstd = tl.rsqrt(_f32_add(variance, eps))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, momentum), _f32_mul(old_mean, 1.0 - momentum))
    new_var = _f32_add(
        _f32_mul(_f32_mul(variance, running_var_correction), momentum),
        _f32_mul(old_var, 1.0 - momentum),
    )

    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)


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


@oracle_impl(
    hardware="B200",
    point="b0183d97",
    BLOCK_K=8192,
    BLOCK_E=1024,
    num_warps_stats=8,
    num_warps_act=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
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

    _bn_stats_kernel[(channels,)](
        arg0_1,
        arg1_1,
        arg2_1,
        mean,
        invstd,
        channels=channels,
        width=width,
        hw_size=hw_size,
        elements_per_channel=elements_per_channel,
        x_stride_n=int(arg0_1.stride(0)),
        x_stride_c=int(arg0_1.stride(1)),
        x_stride_h=int(arg0_1.stride(2)),
        x_stride_w=int(arg0_1.stride(3)),
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_K=BLOCK_K,
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

    return mean, invstd, shuffled[:, :channels, :, :], shuffled[:, channels:, :, :], arg1_1, arg2_1
