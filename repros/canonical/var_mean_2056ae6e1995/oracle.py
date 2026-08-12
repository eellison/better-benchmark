"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 training-BatchNorm channel-cat scope, including bf16-to-fp32 channel statistics over `[N,H,W]`, returned fp32 mean and eps=1e-5 rsqrt side outputs, in-place running mean/variance copy_ updates with the captured fixed correction constant, fp32 affine normalization, final bf16 rounding before ReLU, and direct channels-last bf16 cat output stores, whereas Inductor lowers the captured var_mean/running-stat update/affine/cast/ReLU/cat graph through generic reduction, pointwise, and cat schedules; Inductor cannot do this today because its scheduler does not keep mutable BN side outputs and the following layout-sensitive channel concatenation in one full-scope training-BN plan; the fix is SCHEDULER_FUSION: extend the training BatchNorm template to emit returned statistics, running-stat aliases, cast/ReLU epilogue, and static channel-cat stores directly from the channel-statistics schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bn_partial_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
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
    hw = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(vals, axis=1)
    sums2 = tl.sum(vals * vals, axis=1)
    out_offsets = e_block * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channels < C)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
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
    sums2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(sums, axis=1).to(tl.float32)
    total2 = tl.sum(sums2, axis=1).to(tl.float32)
    mean = total / E
    var = total2 / E - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = libdevice.rsqrt(var + 1.0e-5)
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        old_var * 0.9 + var * 1.0000398612827361 * 0.1,
        mask=channel_mask,
    )
    tl.store(mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)


@triton.jit
def _bn_relu_cat_channels_last_kernel(
    x_ptr,
    skip_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c2 = offsets % (C * 2)
    w_idx = (offsets // (C * 2)) % W
    h_idx = (offsets // (C * 2 * W)) % H
    n_idx = offsets // (C * 2 * H * W)

    first_half = c2 < C
    c = tl.where(first_half, c2, c2 - C)
    input_offsets = n_idx * C * H * W + h_idx * W * C + w_idx * C + c

    skip = tl.load(skip_ptr + input_offsets, mask=mask & first_half, other=0.0)
    x = tl.load(x_ptr + input_offsets, mask=mask & ~first_half, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask & ~first_half, other=0.0)
    invstd = tl.load(invstd_ptr + c, mask=mask & ~first_half, other=0.0)
    weight = tl.load(weight_ptr + c, mask=mask & ~first_half, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask & ~first_half, other=0.0).to(tl.float32)
    affine = ((x - mean) * invstd) * weight + bias
    relu = tl.maximum(affine.to(tl.bfloat16), 0.0)
    value = tl.where(first_half, skip, relu)
    tl.store(out_ptr + offsets, value, mask=mask)


# d3838873: (T([512,480,7,7], bf16), T([480], f32), T([480], f32), T([480], f32), T([480], f32), T([512,480,7,7], bf16))
@oracle_impl(hardware="B200", point="d3838873", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=1024, num_warps=4)
# c9e11f4c: (T([512,336,14,14], bf16), T([336], f32), T([336], f32), T([336], f32), T([336], f32), T([512,336,14,14], bf16))
@oracle_impl(hardware="B200", point="c9e11f4c", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=1024, num_warps=4)
# 02e128f4: (T([512,92,14,14], bf16), T([92], f32), T([92], f32), T([92], f32), T([92], f32), T([512,92,14,14], bf16))
@oracle_impl(hardware="B200", point="02e128f4", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=4, OUT_BLOCK=1024, num_warps=4)
# 2fd52f04: (T([512,100,14,14], bf16), T([100], f32), T([100], f32), T([100], f32), T([100], f32), T([512,100,14,14], bf16))
@oracle_impl(hardware="B200", point="2fd52f04", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=4, OUT_BLOCK=1024, num_warps=4)
# 139bf125: (T([512,120,28,28], bf16), T([120], f32), T([120], f32), T([120], f32), T([120], f32), T([512,120,28,28], bf16))
@oracle_impl(hardware="B200", point="139bf125", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=2, OUT_BLOCK=1024, num_warps=4)
# de73d9ef: (T([512,36,56,56], bf16), T([36], f32), T([36], f32), T([36], f32), T([36], f32), T([512,36,56,56], bf16))
@oracle_impl(hardware="B200", point="de73d9ef", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=1, OUT_BLOCK=1024, num_warps=4)
# 023ff9dc: (T([512,24,112,112], bf16), T([24], f32), T([24], f32), T([24], f32), T([24], f32), T([512,24,112,112], bf16))
@oracle_impl(hardware="B200", point="023ff9dc", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=1, OUT_BLOCK=1024, num_warps=4)
# 8f9cb8b8: (T([512,8,112,112], bf16), T([8], f32), T([8], f32), T([8], f32), T([8], f32), T([512,8,112,112], bf16))
@oracle_impl(hardware="B200", point="8f9cb8b8", BLOCK_E=1024, C_BLOCK=4, FINAL_C_BLOCK=1, OUT_BLOCK=1024, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_E,
    C_BLOCK,
    FINAL_C_BLOCK,
    OUT_BLOCK,
    num_warps,
):
    x, running_mean, running_var, weight, bias, skip = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    e = n * h * w
    num_chunks = triton.cdiv(e, BLOCK_E)
    block_chunks = triton.next_power_of_2(num_chunks)

    partial_sum = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (n, c * 2, h, w),
        (c * 2 * h * w, 1, c * 2 * w, c * 2),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _bn_partial_stats_kernel[(triton.cdiv(c, C_BLOCK), num_chunks)](
        x,
        partial_sum,
        partial_sum2,
        C=c,
        H=h,
        W=w,
        E=e,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _bn_finalize_stats_kernel[(triton.cdiv(c, FINAL_C_BLOCK),)](
        partial_sum,
        partial_sum2,
        running_mean,
        running_var,
        mean,
        invstd,
        C=c,
        E=e,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=8 if block_chunks >= 2048 else 4,
        num_stages=3,
    )
    _bn_relu_cat_channels_last_kernel[(triton.cdiv(out.numel(), OUT_BLOCK),)](
        x,
        skip,
        weight,
        bias,
        mean,
        invstd,
        out,
        C=c,
        H=h,
        W=w,
        TOTAL=out.numel(),
        BLOCK=OUT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return mean, invstd, out, running_mean, running_var
