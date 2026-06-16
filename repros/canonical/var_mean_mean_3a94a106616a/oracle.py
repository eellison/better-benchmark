"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 training-BatchNorm hard-swish scope, including population var_mean over N/H/W, eps=1e-3 rsqrt, mutable running-stat copy_ side effects with the captured correction constant, affine bf16 round trip, hard-swish, final bf16 spatial mean, and the returned `[N,C]` view in compact Triton statistics and epilogue kernels; Inductor lowers the captured var_mean/update/affine/cast/hard-swish/mean graph through separate normalization and reduction regions with extra materialization around the observable dtype boundaries; the fix is SCHEDULER_FUSION: extend the BN-training template to expose saved mean/invstd and running-stat epilogues while sinking fixed-shape affine hard-swish plus spatial-mean consumers into the same channel-statistics lowering."""

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
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets_e = tl.arange(0, BLOCK_E)
    mask = offsets_e < E
    n_idx = offsets_e // HW
    hw_idx = offsets_e - n_idx * HW
    x_offsets = n_idx * C * HW + channel * HW + hw_idx

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x / E
    var = tl.maximum(sum_x2 / E - mean * mean, 0.0)
    invstd = libdevice.rsqrt(var + 0.001)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    updated_mean = mean * 0.01 + old_mean * 0.99
    updated_var = var * 1.0006381620931717 * 0.01 + old_var * 0.99

    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, updated_mean)
    tl.store(running_var_ptr + channel, updated_var)


@triton.jit
def _hardswish_spatial_mean_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    mean_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    hw_offsets = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < HW
    mask = row_mask[:, None] & hw_mask[None, :]
    channels = rows - (rows // C) * C
    offsets = rows[:, None] * HW + hw_offsets[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    normalized = (x - mean[:, None]) * invstd[:, None]
    affine = normalized * weight[:, None] + bias[:, None]
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    relu6 = tl.minimum(tl.maximum(rounded + 3.0, 0.0), 6.0)
    hardswish = rounded * relu6 / 6.0
    hardswish_bf16 = hardswish.to(tl.bfloat16)
    reduced = tl.sum(tl.where(mask, hardswish_bf16.to(tl.float32), 0.0), axis=1)
    mean_value = (reduced / HW).to(tl.bfloat16)
    tl.store(mean_out_ptr + rows, mean_value, mask=row_mask)


# e8a9d13a: (T([32,960,7,7], bf16), T([960], f32), T([960], f32), T([960], f32), T([960], f32), [32,960])
@oracle_impl(hardware="B200", point="e8a9d13a", BLOCK_E=2048, BLOCK_HW=64, ROW_BLOCK=8, STAT_WARPS=2, OUT_WARPS=1)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
    STAT_WARPS: int,
    OUT_WARPS: int,
):
    x, running_mean, running_var, weight, bias, _shape_param_0 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
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

    _bn_stats_kernel[(c,)](
        x,
        running_mean,
        running_var,
        saved_mean,
        invstd,
        C=c,
        HW=hw,
        E=e,
        BLOCK_E=BLOCK_E,
        num_warps=STAT_WARPS,
        num_stages=1,
    )
    _hardswish_spatial_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        saved_mean,
        invstd,
        weight,
        bias,
        spatial_mean,
        C=c,
        HW=hw,
        TOTAL_ROWS=total_rows,
        BLOCK_HW=BLOCK_HW,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=1,
    )
    return saved_mean, invstd, spatial_mean, running_mean, running_var
