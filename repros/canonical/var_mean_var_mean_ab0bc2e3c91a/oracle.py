"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete VisFormer chained bf16 training-BatchNorm scope with channel-blocked Triton reductions, including both population `var_mean(..., [0,2,3], correction=0, keepdim=True)` calls, eps=1e-5 `rsqrt` placements, the first affine bf16 cast before residual add, the returned intermediate activation, final bf16 affine output, and all four mutable running-stat `copy_` aliases, whereas Inductor lowers the captured dual-BN graph through separate generic norm-template and elementwise/layout schedules; Inductor cannot do this today because its training-normalization scheduler does not carry a bf16-cast first-BN epilogue, dependent second reduction, returned side tensors, and mutable copy outputs through one full-scope channelwise plan for NHWC-strided VisFormer tensors; the fix is SCHEDULER_FUSION: extend the BN-training template to fuse chained norm producers and consumers while preserving visible dtype boundaries, returned stats, and running-stat aliases."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _partial_stats_x_kernel(
    x_ptr,
    sum_ptr,
    sumsq_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_c = tl.program_id(1)
    k = pid_k * block_k + tl.arange(0, block_k)[:, None]
    c = pid_c * block_c + tl.arange(0, block_c)[None, :]
    mask = (k < elements_per_channel) & (c < channels)
    x = tl.load(x_ptr + k * channels + c, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    s = tl.sum(x, axis=0)
    ss = tl.sum(_f32_mul(x, x), axis=0)
    cols = pid_c * block_c + tl.arange(0, block_c)
    col_mask = cols < channels
    tl.store(sum_ptr + pid_k * channels + cols, s, mask=col_mask)
    tl.store(sumsq_ptr + pid_k * channels + cols, ss, mask=col_mask)


@triton.jit
def _finalize_first_stats_kernel(
    sum_ptr,
    sumsq_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_blocks: tl.constexpr,
    eps: tl.constexpr,
    running_var_correction: tl.constexpr,
    block_p: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_c = tl.program_id(0)
    p = tl.arange(0, block_p)[:, None]
    c = pid_c * block_c + tl.arange(0, block_c)[None, :]
    mask = (p < num_blocks) & (c < channels)
    partial_sum = tl.load(sum_ptr + p * channels + c, mask=mask, other=0.0)
    partial_sumsq = tl.load(sumsq_ptr + p * channels + c, mask=mask, other=0.0)
    total = tl.sum(partial_sum, axis=0)
    total_sumsq = tl.sum(partial_sumsq, axis=0)
    inv_count = 1.0 / elements_per_channel
    mean = _f32_mul(total, inv_count)
    mean_sq = _f32_mul(mean, mean)
    var = _f32_add(_f32_mul(total_sumsq, inv_count), -mean_sq)
    var = tl.maximum(var, 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, eps))

    cols = pid_c * block_c + tl.arange(0, block_c)
    col_mask = cols < channels
    old_mean = tl.load(running_mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    var_for_running = _f32_mul(_f32_mul(var, running_var_correction), 0.1)
    new_var = _f32_add(var_for_running, _f32_mul(old_var, 0.9))
    tl.store(running_mean_ptr + cols, new_mean, mask=col_mask)
    tl.store(running_var_ptr + cols, new_var, mask=col_mask)
    tl.store(mean_ptr + cols, mean, mask=col_mask)
    tl.store(invstd_ptr + cols, invstd, mask=col_mask)


@triton.jit
def _first_affine_second_partial_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    add4_ptr,
    sum_ptr,
    sumsq_ptr,
    channels: tl.constexpr,
    hw_size: tl.constexpr,
    elements_per_channel: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_c = tl.program_id(1)
    k = pid_k * block_k + tl.arange(0, block_k)[:, None]
    c = pid_c * block_c + tl.arange(0, block_c)[None, :]
    mask = (k < elements_per_channel) & (c < channels)
    hw = k % hw_size
    x_offsets = k * channels + c
    residual_offsets = hw * channels + c

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c < channels, other=0.0).to(tl.float32)

    centered = _f32_add(x, -mean)
    normed = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normed, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16).to(tl.float32)
    add4 = _f32_add(affine_bf16, residual)
    add4 = tl.where(mask, add4, 0.0)
    tl.store(add4_ptr + x_offsets, add4, mask=mask)

    s = tl.sum(add4, axis=0)
    ss = tl.sum(_f32_mul(add4, add4), axis=0)
    cols = pid_c * block_c + tl.arange(0, block_c)
    col_mask = cols < channels
    tl.store(sum_ptr + pid_k * channels + cols, s, mask=col_mask)
    tl.store(sumsq_ptr + pid_k * channels + cols, ss, mask=col_mask)


@triton.jit
def _finalize_second_stats_kernel(
    sum_ptr,
    sumsq_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_ptr,
    mean_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    num_blocks: tl.constexpr,
    eps: tl.constexpr,
    running_var_correction: tl.constexpr,
    block_p: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_c = tl.program_id(0)
    p = tl.arange(0, block_p)[:, None]
    c = pid_c * block_c + tl.arange(0, block_c)[None, :]
    mask = (p < num_blocks) & (c < channels)
    partial_sum = tl.load(sum_ptr + p * channels + c, mask=mask, other=0.0)
    partial_sumsq = tl.load(sumsq_ptr + p * channels + c, mask=mask, other=0.0)
    total = tl.sum(partial_sum, axis=0)
    total_sumsq = tl.sum(partial_sumsq, axis=0)
    inv_count = 1.0 / elements_per_channel
    mean = _f32_mul(total, inv_count)
    mean_sq = _f32_mul(mean, mean)
    var = _f32_add(_f32_mul(total_sumsq, inv_count), -mean_sq)
    var = tl.maximum(var, 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, eps))

    cols = pid_c * block_c + tl.arange(0, block_c)
    col_mask = cols < channels
    old_mean = tl.load(running_mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    var_for_running = _f32_mul(_f32_mul(var, running_var_correction), 0.1)
    new_var = _f32_add(var_for_running, _f32_mul(old_var, 0.9))
    tl.store(running_mean_ptr + cols, new_mean, mask=col_mask)
    tl.store(running_var_ptr + cols, new_var, mask=col_mask)
    tl.store(invstd_ptr + cols, invstd, mask=col_mask)
    tl.store(mean_ptr + cols, mean, mask=col_mask)


@triton.jit
def _second_affine_kernel(
    add4_ptr,
    weight_ptr,
    bias_ptr,
    invstd_ptr,
    mean_ptr,
    out_ptr,
    channels: tl.constexpr,
    elements_per_channel: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_c = tl.program_id(1)
    k = pid_k * block_k + tl.arange(0, block_k)[:, None]
    c = pid_c * block_c + tl.arange(0, block_c)[None, :]
    mask = (k < elements_per_channel) & (c < channels)
    offsets = k * channels + c
    x = tl.load(add4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
    centered = _f32_add(x, -mean)
    normed = _f32_mul(centered, invstd)
    y = _f32_add(_f32_mul(normed, weight), bias).to(tl.bfloat16)
    tl.store(out_ptr + offsets, y, mask=mask)


# c9bb6ab0: (T([128,768,7,7], bf16, stride=(37632,1,5376,768)), T([768], f32), ..., T([1,768,7,7], f32, stride=(37632,1,5376,768)), ...)
@oracle_impl(hardware="B200", point="c9bb6ab0", CHANNELS=768, HEIGHT=7, WIDTH=7, ELEMENTS_PER_CHANNEL=6272, NUM_BLOCKS=25, BLOCK_P=32, BLOCK_K=256, BLOCK_C=32, num_warps=8)
# 7b85be27: (T([128,384,14,14], bf16, stride=(75264,1,5376,384)), T([384], f32), ..., T([1,384,14,14], f32, stride=(75264,1,5376,384)), ...)
@oracle_impl(hardware="B200", point="7b85be27", CHANNELS=384, HEIGHT=14, WIDTH=14, ELEMENTS_PER_CHANNEL=25088, NUM_BLOCKS=49, BLOCK_P=64, BLOCK_K=512, BLOCK_C=16, num_warps=8)
# ce0e5e3c: (T([128,192,28,28], bf16, stride=(150528,1,5376,192)), T([192], f32), ..., T([1,192,28,28], f32, stride=(150528,1,5376,192)), ...)
@oracle_impl(hardware="B200", point="ce0e5e3c", CHANNELS=192, HEIGHT=28, WIDTH=28, ELEMENTS_PER_CHANNEL=100352, NUM_BLOCKS=196, BLOCK_P=256, BLOCK_K=512, BLOCK_C=16, num_warps=8)
def oracle_forward(
    inputs,
    *,
    CHANNELS: int,
    HEIGHT: int,
    WIDTH: int,
    ELEMENTS_PER_CHANNEL: int,
    NUM_BLOCKS: int,
    BLOCK_P: int,
    BLOCK_K: int,
    BLOCK_C: int,
    num_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
    ) = inputs

    hw_size = HEIGHT * WIDTH
    activation_shape = (128, CHANNELS, HEIGHT, WIDTH)
    activation_stride = (CHANNELS * hw_size, 1, WIDTH * CHANNELS, CHANNELS)
    stat4_shape = (1, CHANNELS, 1, 1)
    stat4_stride = (CHANNELS, 1, 1, 1)

    sum1 = torch.empty((NUM_BLOCKS, CHANNELS), device=arg0_1.device, dtype=torch.float32)
    sumsq1 = torch.empty((NUM_BLOCKS, CHANNELS), device=arg0_1.device, dtype=torch.float32)
    sum2 = torch.empty((NUM_BLOCKS, CHANNELS), device=arg0_1.device, dtype=torch.float32)
    sumsq2 = torch.empty((NUM_BLOCKS, CHANNELS), device=arg0_1.device, dtype=torch.float32)
    mean1 = torch.empty_strided(stat4_shape, stat4_stride, device=arg0_1.device, dtype=torch.float32)
    invstd1 = torch.empty_strided(stat4_shape, stat4_stride, device=arg0_1.device, dtype=torch.float32)
    add4 = torch.empty_strided(activation_shape, activation_stride, device=arg0_1.device, dtype=torch.float32)
    invstd2 = torch.empty((CHANNELS,), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(activation_shape, activation_stride, device=arg0_1.device, dtype=torch.bfloat16)
    mean2 = torch.empty_strided(stat4_shape, stat4_stride, device=arg0_1.device, dtype=torch.float32)

    grid = (NUM_BLOCKS, triton.cdiv(CHANNELS, BLOCK_C))
    _partial_stats_x_kernel[grid](
        arg0_1,
        sum1,
        sumsq1,
        CHANNELS,
        ELEMENTS_PER_CHANNEL,
        BLOCK_K,
        BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_first_stats_kernel[(triton.cdiv(CHANNELS, BLOCK_C),)](
        sum1,
        sumsq1,
        arg1_1,
        arg2_1,
        mean1,
        invstd1,
        CHANNELS,
        ELEMENTS_PER_CHANNEL,
        NUM_BLOCKS,
        1.0e-5,
        1.0001594642002871,
        BLOCK_P,
        BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _first_affine_second_partial_kernel[grid](
        arg0_1,
        arg5_1,
        arg3_1,
        arg4_1,
        mean1,
        invstd1,
        add4,
        sum2,
        sumsq2,
        CHANNELS,
        hw_size,
        ELEMENTS_PER_CHANNEL,
        BLOCK_K,
        BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_second_stats_kernel[(triton.cdiv(CHANNELS, BLOCK_C),)](
        sum2,
        sumsq2,
        arg6_1,
        arg7_1,
        invstd2,
        mean2,
        CHANNELS,
        ELEMENTS_PER_CHANNEL,
        NUM_BLOCKS,
        1.0e-5,
        1.0001594642002871,
        BLOCK_P,
        BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _second_affine_kernel[grid](
        add4,
        arg8_1,
        arg9_1,
        invstd2,
        mean2,
        out,
        CHANNELS,
        ELEMENTS_PER_CHANNEL,
        BLOCK_K,
        BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return mean1, invstd1, add4, invstd2, out, mean2, arg1_1, arg2_1, arg6_1, arg7_1
