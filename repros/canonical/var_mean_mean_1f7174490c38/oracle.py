"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete channels-last MobileNetV2 bf16 training-BatchNorm ReLU6 scope, including population var_mean over N/H/W, eps=1e-5 rsqrt, mutable running-stat copy_ side effects with the captured correction constant, affine bf16 round trip, ReLU6 bf16 round trip, final bf16 spatial mean, and the returned `[N,C]` view in compact Triton statistics and epilogue kernels; Inductor lowers the captured var_mean/update/affine/cast/ReLU6/mean graph through separate normalization and reduction regions with extra materialization around observable dtype and view boundaries; the fix is SCHEDULER_FUSION: extend the BN-training template to expose saved mean/invstd and running-stat epilogues while sinking fixed-shape affine ReLU6 plus spatial-mean consumers into the same channels-last lowering."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bn_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    saved_mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.arange(0, BLOCK_R)
    hw = r_offsets % (H * W)
    n_idx = r_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    mask = (channels[:, None] < C) & (r_offsets[None, :] < E)
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )

    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    vals = tl.where(mask, vals, 0.0)
    sums = tl.sum(vals, axis=1)
    sums2 = tl.sum(vals * vals, axis=1)
    mean = sums / E
    var = tl.maximum(sums2 / E - mean * mean, 0.0)
    invstd = libdevice.rsqrt(var + 1.0e-5)

    channel_mask = channels < C
    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    new_mean = mean * 0.1 + old_mean * 0.9
    new_var = var * 1.0001594642002871 * 0.1 + old_var * 0.9

    tl.store(saved_mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, new_var, mask=channel_mask)


@triton.jit
def _relu6_spatial_mean_kernel(
    x_ptr,
    saved_mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    hw_offsets = tl.arange(0, BLOCK_HW)
    channels = rows - (rows // C) * C
    n_idx = rows // C
    h_idx = hw_offsets // W
    w_idx = hw_offsets - h_idx * W
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < (H * W)
    mask = row_mask[:, None] & hw_mask[None, :]
    x_offsets = (
        n_idx[:, None] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(saved_mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    normalized = (x - mean[:, None]) * invstd[:, None]
    affine = normalized * weight[:, None] + bias[:, None]
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    relu6 = tl.minimum(tl.maximum(rounded, 0.0), 6.0).to(tl.bfloat16)
    reduced = tl.sum(tl.where(mask, relu6.to(tl.float32), 0.0), axis=1)
    mean_value = (reduced / (H * W)).to(tl.bfloat16)
    tl.store(spatial_mean_ptr + rows, mean_value, mask=row_mask)


# 86b98b1a: (T([128,1280,7,7], bf16, stride=[62720,1,8960,1280]), T([1280], f32), T([1280], f32), T([1280], f32), T([1280], f32), ...)
@oracle_impl(hardware="B200", point="86b98b1a", BLOCK_R=8192, BLOCK_C=16, BLOCK_HW=64, ROW_BLOCK=64, STAT_WARPS=8, OUT_WARPS=1)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
    STAT_WARPS: int,
    OUT_WARPS: int,
):
    x, running_mean, running_var, weight, bias, _shape0, _stride0, _shape1 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    elements_per_channel = n * h * w
    total_rows = n * c

    saved_mean = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    spatial_mean = torch.empty_strided(
        (n, c),
        (c, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _bn_stats_kernel[(triton.cdiv(c, BLOCK_C),)](
        x,
        running_mean,
        running_var,
        saved_mean,
        invstd,
        C=c,
        H=h,
        W=w,
        E=elements_per_channel,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=STAT_WARPS,
        num_stages=3,
    )
    _relu6_spatial_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        saved_mean,
        invstd,
        weight,
        bias,
        spatial_mean,
        C=c,
        H=h,
        W=w,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        TOTAL_ROWS=total_rows,
        BLOCK_HW=BLOCK_HW,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=1,
    )
    return saved_mean, invstd, spatial_mean, running_mean, running_var
