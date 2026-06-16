"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle folds the fixed 2x2 stride-2 avg_pool2d producer into the bf16 training-BatchNorm var_mean/update/affine/ReLU scope, returns the materialized pooled tensor plus invstd and saved-mean side outputs, and performs the captured running mean/variance copy_ updates from the same channel-statistics plan, whereas Inductor schedules the structured pooling stencil, normalization reduction, mutable running-stat epilogues, bf16 cast boundary, and activation consumer as generic separated work; Inductor cannot do this today because scheduler fusion does not inline a pooling producer into a multi-output normalization reduction with side-effecting copy_ returns and a full activation epilogue; the fix is SCHEDULER_FUSION: extend the BN-training norm template to accept fixed-window pooling producers and emit pooled output, stats side outputs, running-stat aliases, bf16 cast, and ReLU from one full-scope schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _pool_bn_single_pass_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    pooled_ptr,
    invstd_ptr,
    relu_ptr,
    mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    channel = tl.program_id(0)
    e_offsets = tl.arange(0, BLOCK_E)
    mask = e_offsets < E
    out_hw = OH * OW
    n_idx = e_offsets // out_hw
    out_hw_idx = e_offsets - n_idx * out_hw
    out_h = out_hw_idx // OW
    out_w = out_hw_idx - out_h * OW
    input_base = (
        n_idx * stride_n
        + channel * stride_c
        + (out_h * 2) * stride_h
        + (out_w * 2) * stride_w
    )

    x00 = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(x_ptr + input_base + stride_w, mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + input_base + stride_h, mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + input_base + stride_h + stride_w, mask=mask, other=0.0).to(tl.float32)
    pooled_f32 = (x00 + x01 + x10 + x11) * 0.25
    pooled = pooled_f32.to(tl.bfloat16)

    pooled_offsets = n_idx * C * out_hw + channel * out_hw + out_hw_idx
    tl.store(pooled_ptr + pooled_offsets, pooled, mask=mask)

    sum_x = tl.sum(pooled_f32, axis=0)
    mean = sum_x / E
    diff = tl.where(mask, pooled_f32 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / E
    invstd = tl.rsqrt(var + 1.0e-5)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    correction = 1.005128205128205
    tl.store(running_mean_ptr + channel, old_mean * 0.9 + mean * 0.1)
    tl.store(running_var_ptr + channel, old_var * 0.9 + var * correction * 0.1)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(mean_ptr + channel, mean)

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    y = ((pooled_f32 - mean) * invstd) * weight + bias
    y = y.to(tl.bfloat16)
    y = tl.where(y != y, y, tl.maximum(y, 0.0))
    tl.store(relu_ptr + pooled_offsets, y, mask=mask)


@triton.jit
def _pool_bn_partial_stats_kernel(
    x_ptr,
    pooled_ptr,
    partial_sum_ptr,
    partial_scratch_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
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
    out_hw = OH * OW
    n_idx = e_offsets // out_hw
    out_hw_idx = e_offsets - n_idx * out_hw
    out_h = out_hw_idx // OW
    out_w = out_hw_idx - out_h * OW
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    input_base = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + (out_h[None, :] * 2) * stride_h
        + (out_w[None, :] * 2) * stride_w
    )

    x00 = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(x_ptr + input_base + stride_w, mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + input_base + stride_h, mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + input_base + stride_h + stride_w, mask=mask, other=0.0).to(tl.float32)
    pooled_f32 = (x00 + x01 + x10 + x11) * 0.25
    pooled = pooled_f32.to(tl.bfloat16)

    pooled_offsets = n_idx[None, :] * C * out_hw + channels[:, None] * out_hw + out_hw_idx[None, :]
    tl.store(pooled_ptr + pooled_offsets, pooled, mask=mask)

    sums = tl.sum(pooled_f32, axis=1)
    partial_offsets = e_block * C + channels
    tl.store(partial_sum_ptr + partial_offsets, sums, mask=channels < C)


@triton.jit
def _pool_bn_finalize_mean_kernel(
    partial_sum_ptr,
    mean_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(sums, axis=1).to(tl.float32)
    mean = total / E
    channel_mask = channels < C
    tl.store(mean_ptr + channels, mean, mask=channel_mask)


@triton.jit
def _pool_bn_partial_m2_kernel(
    x_ptr,
    mean_ptr,
    partial_m2_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
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
    out_hw = OH * OW
    n_idx = e_offsets // out_hw
    out_hw_idx = e_offsets - n_idx * out_hw
    out_h = out_hw_idx // OW
    out_w = out_hw_idx - out_h * OW
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    input_base = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + (out_h[None, :] * 2) * stride_h
        + (out_w[None, :] * 2) * stride_w
    )
    x00 = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(x_ptr + input_base + stride_w, mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + input_base + stride_h, mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + input_base + stride_h + stride_w, mask=mask, other=0.0).to(tl.float32)
    pooled = (x00 + x01 + x10 + x11) * 0.25
    mean = tl.load(mean_ptr + channels, mask=channels < C, other=0.0).to(tl.float32)
    diff = tl.where(mask, pooled - mean[:, None], 0.0)
    m2 = tl.sum(diff * diff, axis=1)
    partial_offsets = e_block * C + channels
    tl.store(partial_m2_ptr + partial_offsets, m2, mask=channels < C)


@triton.jit
def _pool_bn_finalize_var_kernel(
    partial_m2_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_ptr,
    mean_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    m2 = tl.load(partial_m2_ptr + offsets, mask=mask, other=0.0)
    total_m2 = tl.sum(m2, axis=1).to(tl.float32)
    var = total_m2 / E
    invstd = tl.rsqrt(var + 1.0e-5)
    correction = 1.005128205128205
    channel_mask = channels < C
    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(running_var_ptr + channels, old_var * 0.9 + var * correction * 0.1, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)


@triton.jit
def _pool_bn_affine_relu_kernel(
    pooled_ptr,
    weight_ptr,
    bias_ptr,
    invstd_ptr,
    mean_ptr,
    relu_ptr,
    C: tl.constexpr,
    OUT_HW: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channels = (offsets // OUT_HW) % C
    pooled = tl.load(pooled_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    y = ((pooled - mean) * invstd) * weight + bias
    y = y.to(tl.bfloat16)
    y = tl.where(y != y, y, tl.maximum(y, 0.0))
    tl.store(relu_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="30ad209f", BLOCK_E=256, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=1)
@oracle_impl(hardware="B200", point="fe4b2426", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=4)
@oracle_impl(hardware="B200", point="04752777", BLOCK_E=4096, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=8)
@oracle_impl(hardware="B200", point="00fa08bd", BLOCK_E=2048, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=True, num_warps=4)
@oracle_impl(hardware="B200", point="e4ee805e", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=False, num_warps=4)
@oracle_impl(hardware="B200", point="3637c02c", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=16, USE_SINGLE=False, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    FINAL_C_BLOCK: int,
    USE_SINGLE: bool,
    num_warps: int,
):
    x, running_mean, running_var, weight, bias = inputs
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    oh = h // 2
    ow = w // 2
    out_hw = oh * ow
    e = n * out_hw
    total = n * c * out_hw

    pooled = torch.empty_strided((n, c, oh, ow), (c * out_hw, out_hw, ow, 1), device=x.device, dtype=torch.bfloat16)
    invstd = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    relu = torch.empty_strided((n, c, oh, ow), (c * out_hw, out_hw, ow, 1), device=x.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)

    if USE_SINGLE:
        _pool_bn_single_pass_kernel[(c,)](
            x,
            running_mean,
            running_var,
            weight,
            bias,
            pooled,
            invstd,
            relu,
            mean,
            c,
            h,
            w,
            oh,
            ow,
            e,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_E=BLOCK_E,
            num_warps=num_warps,
        )
        return pooled, invstd, relu, mean, running_mean, running_var

    num_chunks = triton.cdiv(e, BLOCK_E)
    block_chunks = triton.next_power_of_2(num_chunks)
    partial_sum = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_m2 = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)

    _pool_bn_partial_stats_kernel[(triton.cdiv(c, C_BLOCK), num_chunks)](
        x,
        pooled,
        partial_sum,
        partial_m2,
        c,
        h,
        w,
        oh,
        ow,
        e,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=num_warps,
    )
    _pool_bn_finalize_mean_kernel[(triton.cdiv(c, FINAL_C_BLOCK),)](
        partial_sum,
        mean,
        c,
        e,
        num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=4,
    )
    _pool_bn_partial_m2_kernel[(triton.cdiv(c, C_BLOCK), num_chunks)](
        x,
        mean,
        partial_m2,
        c,
        h,
        w,
        oh,
        ow,
        e,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=num_warps,
    )
    _pool_bn_finalize_var_kernel[(triton.cdiv(c, FINAL_C_BLOCK),)](
        partial_m2,
        running_mean,
        running_var,
        invstd,
        mean,
        c,
        e,
        num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=4,
    )
    _pool_bn_affine_relu_kernel[(triton.cdiv(total, 1024),)](
        pooled,
        weight,
        bias,
        invstd,
        mean,
        relu,
        c,
        out_hw,
        total,
        BLOCK=1024,
        num_warps=4,
    )
    return pooled, invstd, relu, mean, running_mean, running_var
