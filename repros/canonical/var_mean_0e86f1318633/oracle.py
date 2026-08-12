"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Adv-Inception training-BatchNorm var_mean scope, including fp32 population Welford statistics over channels-last input, eps=0.001 rsqrt side output, mutable running-stat copy_ aliases with the captured correction literal, exact fp32 affine followed by the explicit bf16 cast/ReLU boundary, and returned 3x3 stride-2 low-memory maxpool values plus int8 offsets, whereas Inductor lowers the statistics, running-stat epilogues, rounded activation materialization, and maxpool-with-offsets consumer as separate generic schedules; Inductor cannot do this today because scheduler fusion does not keep training-BN reductions with visible mutable aliases virtual through a downstream low-memory pooling stencil while preserving bf16 conversion and offset semantics; the fix is SCHEDULER_FUSION: extend the training-BatchNorm scheduler to emit running-stat side effects and sink the exact bf16 affine/ReLU producer into fixed maxpool-with-offsets stencils."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 64
H_IN = 147
W_IN = 147
H_OUT = 73
W_OUT = 73
IN_HW = H_IN * W_IN
OUT_HW = H_OUT * W_OUT
ELEM_PER_CHANNEL = BATCH * IN_HW
RUNNING_VAR_CORRECTION = 1.0000003615393043


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
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _bn_partial_stats_kernel(
    x_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    e_block = tl.program_id(1)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = e_block * BLOCK_E + tl.arange(0, BLOCK_E)

    hw_idx = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    zero = tl.zeros((C_BLOCK, BLOCK_E), tl.float32)
    weights = tl.where(mask, 1.0, 0.0)
    block_mean, block_m2, block_weight = triton_helpers.welford(vals, zero, weights, 1)

    out_offsets = e_block * C + channels
    channel_mask = channels < C
    tl.store(partial_mean_ptr + out_offsets, block_mean, mask=channel_mask)
    tl.store(partial_m2_ptr + out_offsets, block_m2, mask=channel_mask)
    tl.store(partial_weight_ptr + out_offsets, block_weight, mask=channel_mask)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    CORRECTION: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]

    block_mean = tl.load(partial_mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_m2 = tl.load(partial_m2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_weight = tl.load(partial_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean, m2, _weight = triton_helpers.welford(block_mean, block_m2, block_weight, 1)
    var = m2 / E
    invstd = libdevice.rsqrt(_f32_add(var, 0.001))

    channel_mask = channels < C
    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    updated_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    updated_var = _f32_add(
        _f32_mul(_f32_mul(var, CORRECTION), 0.1),
        _f32_mul(old_var, 0.9),
    )

    tl.store(mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, updated_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, updated_var, mask=channel_mask)


@triton.jit
def _bn_relu_maxpool3_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    values_ptr,
    offsets_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    n_idx = tl.program_id(0)
    c_block = tl.program_id(1)
    out_block = tl.program_id(2)

    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    channel_mask = channels < C
    out_mask = out_offsets < (OUT_H * OUT_W)
    out_h = out_offsets // OUT_W
    out_w = out_offsets - out_h * OUT_W

    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    best = tl.full((BLOCK_OUT, BLOCK_C), -float("inf"), tl.float32)
    best_offset = tl.zeros((BLOCK_OUT, BLOCK_C), tl.int32)

    for kh in tl.static_range(0, 3):
        in_h = out_h * 2 + kh
        for kw in tl.static_range(0, 3):
            in_w = out_w * 2 + kw
            x_offsets = (
                n_idx * stride_n
                + channels[None, :] * stride_c
                + in_h[:, None] * stride_h
                + in_w[:, None] * stride_w
            )
            mask = out_mask[:, None] & channel_mask[None, :]
            x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
            centered = _f32_sub(x, mean[None, :])
            normalized = _f32_mul(centered, invstd[None, :])
            scaled = _f32_mul(normalized, weight[None, :])
            affine = _f32_add(scaled, bias[None, :])
            relu = _relu_preserve_nan(affine.to(tl.bfloat16).to(tl.float32))
            take = mask & ((relu > best) | ((relu != relu) & (best == best)))
            best = tl.where(take, relu, best)
            best_offset = tl.where(take, kh * 3 + kw, best_offset)

    store_offsets = n_idx * (C * OUT_H * OUT_W) + out_offsets[:, None] * C + channels[None, :]
    store_mask = out_mask[:, None] & channel_mask[None, :]
    tl.store(values_ptr + store_offsets, best, mask=store_mask)
    tl.store(offsets_ptr + store_offsets, best_offset.to(tl.int8), mask=store_mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="22c184d8", BLOCK_E=4096, C_BLOCK=4, POOL_BLOCK_C=64, POOL_BLOCK_OUT=16)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    POOL_BLOCK_C: int,
    POOL_BLOCK_OUT: int,
):
    x, running_mean, running_var, weight, bias, _kernel_size, _stride = inputs
    num_chunks = triton.cdiv(ELEM_PER_CHANNEL, BLOCK_E)
    block_chunks = _next_power_of_2(num_chunks)

    partial_mean = torch.empty((num_chunks, CHANNELS), device=x.device, dtype=torch.float32)
    partial_m2 = torch.empty((num_chunks, CHANNELS), device=x.device, dtype=torch.float32)
    partial_weight = torch.empty((num_chunks, CHANNELS), device=x.device, dtype=torch.float32)
    mean = torch.empty_strided((1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1), device=x.device, dtype=torch.float32)
    out_stride = (CHANNELS * OUT_HW, 1, W_OUT * CHANNELS, CHANNELS)
    values = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        out_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    offsets = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        out_stride,
        device=x.device,
        dtype=torch.int8,
    )

    _bn_partial_stats_kernel[(triton.cdiv(CHANNELS, C_BLOCK), num_chunks)](
        x,
        partial_mean,
        partial_m2,
        partial_weight,
        C=CHANNELS,
        H=H_IN,
        W=W_IN,
        E=ELEM_PER_CHANNEL,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _bn_finalize_stats_kernel[(triton.cdiv(CHANNELS, C_BLOCK),)](
        partial_mean,
        partial_m2,
        partial_weight,
        running_mean,
        running_var,
        mean,
        invstd,
        C=CHANNELS,
        E=ELEM_PER_CHANNEL,
        CORRECTION=RUNNING_VAR_CORRECTION,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=C_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _bn_relu_maxpool3_kernel[
        (BATCH, triton.cdiv(CHANNELS, POOL_BLOCK_C), triton.cdiv(OUT_HW, POOL_BLOCK_OUT))
    ](
        x,
        mean,
        invstd,
        weight,
        bias,
        values,
        offsets,
        C=CHANNELS,
        H=H_IN,
        W=W_IN,
        OUT_H=H_OUT,
        OUT_W=W_OUT,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_C=POOL_BLOCK_C,
        BLOCK_OUT=POOL_BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )
    return mean, invstd, values, offsets, running_mean, running_var
