"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete EfficientNet bf16 training-BatchNorm SiLU pad scope, including fp32 population var_mean over N,H,W, eps=0.001 rsqrt side output, running mean/variance copy_ side effects with the captured correction literal, affine scale/bias, the explicit bf16 round trip before natural-exp SiLU, final bf16 left/top/right/bottom zero-pad, and all five returned outputs, whereas Inductor lowers the decomposed var_mean, mutable running-stat updates, affine, SiLU, casts, and pad through generic reduction and pointwise schedules; Inductor cannot do this today because its training-BatchNorm lowering does not select a cooperative split-K channel-statistics template that feeds observable stats, mutable running-stat returns, and a padded activation epilogue while preserving bf16 rounding boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-training schedule that splits the N*H*W reduction, finalizes running-stat side effects, and emits the exact padded SiLU epilogue from the channel-statistics plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-3
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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _partial_sums_kernel(
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
    r_offsets = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    c_mask = c_offsets < channels
    r_mask = r_offsets < elements_per_channel

    n_idx = r_offsets // hw_size
    hw = r_offsets - n_idx * hw_size
    h_idx = hw // width
    w_idx = hw - h_idx * width
    mask = c_mask[:, None] & r_mask[None, :]

    x_offsets = (
        n_idx[None, :] * x_stride_n
        + c_offsets[:, None] * x_stride_c
        + h_idx[None, :] * x_stride_h
        + w_idx[None, :] * x_stride_w
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(_f32_mul(x, x), axis=1)

    out_offsets = tl.program_id(1) * channels + c_offsets
    plane_stride = num_chunks * channels
    tl.store(partial_ptr + out_offsets, sum_x, mask=c_mask)
    tl.store(partial_ptr + plane_stride + out_offsets, sum_x2, mask=c_mask)


@triton.jit
def _finalize_sums_kernel(
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
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    c_mask = c_offsets < channels
    mask = c_mask[:, None] & (chunks[None, :] < num_chunks)
    offsets = chunks[None, :] * channels + c_offsets[:, None]
    plane_stride = num_chunks * channels

    sum_x = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=1)
    sum_x2 = tl.sum(
        tl.load(partial_ptr + plane_stride + offsets, mask=mask, other=0.0),
        axis=1,
    )
    inv_count = 1.0 / elements_per_channel
    mean = _f32_mul(sum_x, inv_count)
    variance = _f32_sub(_f32_mul(sum_x2, inv_count), _f32_mul(mean, mean))
    variance = tl.where(variance < 0.0, 0.0, variance)
    invstd = libdevice.rsqrt(_f32_add(variance, eps))

    old_mean = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0).to(
        tl.float32
    )
    old_var = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=0.0).to(
        tl.float32
    )
    new_mean = _f32_add(_f32_mul(mean, momentum), _f32_mul(old_mean, 1.0 - momentum))
    new_var = _f32_add(
        _f32_mul(_f32_mul(variance, running_var_correction), momentum),
        _f32_mul(old_var, 1.0 - momentum),
    )

    tl.store(mean_ptr + c_offsets, mean, mask=c_mask)
    tl.store(invstd_ptr + c_offsets, invstd, mask=c_mask)
    tl.store(running_mean_ptr + c_offsets, new_mean, mask=c_mask)
    tl.store(running_var_ptr + c_offsets, new_var, mask=c_mask)


@triton.jit
def _bn_silu_pad_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    total_output: tl.constexpr,
    channels: tl.constexpr,
    in_height: tl.constexpr,
    in_width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < total_output

    channel = offsets - (offsets // channels) * channels
    spatial = offsets // channels
    out_w = spatial - (spatial // out_width) * out_width
    out_h = (spatial // out_width) - ((spatial // out_width) // out_height) * out_height
    n_idx = spatial // (out_height * out_width)
    in_h = out_h - 1
    in_w = out_w - 1
    valid_input = mask & (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width)

    x_offsets = (
        n_idx * x_stride_n
        + channel * x_stride_c
        + in_h * x_stride_h
        + in_w * x_stride_w
    )
    x = tl.load(x_ptr + x_offsets, mask=valid_input, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    norm = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(norm, weight), bias)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    denom = _f32_add(libdevice.exp(_f32_mul(rounded, -1.0)), 1.0)
    silu = _f32_div(rounded, denom)
    out = tl.where(
        valid_input,
        silu.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        0.0,
    )
    tl.store(out_ptr + offsets, out, mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="e41460a6",
    BLOCK_R=4096,
    BLOCK_C=8,
    BLOCK_E=512,
    num_warps_stats=8,
    num_warps_final=1,
    num_warps_act=4,
)
@oracle_impl(
    hardware="B200",
    point="46238279",
    BLOCK_R=4096,
    BLOCK_C=8,
    BLOCK_E=512,
    num_warps_stats=8,
    num_warps_final=1,
    num_warps_act=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_E: int,
    num_warps_stats: int,
    num_warps_final: int,
    num_warps_act: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    out_height = height + 3
    out_width = width + 3
    hw_size = height * width
    out_hw_size = out_height * out_width
    elements_per_channel = batch * hw_size
    total_output = batch * channels * out_hw_size
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
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
    padded = torch.empty_strided(
        (batch, channels, out_height, out_width),
        (channels * out_hw_size, 1, out_width * channels, channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (2, num_chunks, channels),
        (num_chunks * channels, channels, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _partial_sums_kernel[(triton.cdiv(channels, BLOCK_C), num_chunks)](
        arg0_1,
        partial,
        channels=channels,
        width=width,
        hw_size=hw_size,
        elements_per_channel=elements_per_channel,
        x_stride_n=int(arg0_1.stride(0)),
        x_stride_c=int(arg0_1.stride(1)),
        x_stride_h=int(arg0_1.stride(2)),
        x_stride_w=int(arg0_1.stride(3)),
        num_chunks=num_chunks,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _finalize_sums_kernel[(triton.cdiv(channels, BLOCK_C),)](
        partial,
        arg1_1,
        arg2_1,
        mean,
        invstd,
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
    _bn_silu_pad_kernel[(triton.cdiv(total_output, BLOCK_E),)](
        arg0_1,
        arg3_1,
        arg4_1,
        mean,
        invstd,
        padded,
        total_output=total_output,
        channels=channels,
        in_height=height,
        in_width=width,
        out_height=out_height,
        out_width=out_width,
        x_stride_n=int(arg0_1.stride(0)),
        x_stride_c=int(arg0_1.stride(1)),
        x_stride_h=int(arg0_1.stride(2)),
        x_stride_w=int(arg0_1.stride(3)),
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_act,
        num_stages=3,
    )

    return mean, invstd, padded, arg1_1, arg2_1
