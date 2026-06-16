"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXt bf16/fp32 channel LayerNorm training scope in one Triton row kernel, including the NHWC view over channels-last NCHW storage, bf16-to-fp32 input conversion, returned fp32 mean from `var_mean(..., dim=3, correction=0, keepdim=True)`, returned fp32 eps=1e-6 rsqrt, fp32 scale/bias affine, final bf16 rounding after the logical NCHW permute, and the direct channels-last NCHW output store, whereas Inductor lowers the captured permute/var_mean/affine/permute/cast graph through generic normalization and layout pointwise scheduling; Inductor cannot do this today because its normalization scheduler does not keep the visible statistic side outputs, affine epilogue, and final layout/cast boundary in one channel-layernorm schedule for this NHWC-over-NCHW pattern; the fix is SCHEDULER_FUSION: add a guarded channel-layernorm schedule that emits returned statistics and the rounded affine store directly for bf16 ConvNeXt channel-last points."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _channel_layernorm_stats_bf16_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rsqrt_ptr,
    out_ptr,
    C: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channels = tl.arange(0, BLOCK_C)[None, :]
    mask = (rows < TOTAL_ROWS) & (channels < C)
    offsets = rows * C + channels

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_values = tl.sum(tl.where(mask, values, 0.0), axis=1)[:, None]
    sum_squares = tl.sum(tl.where(mask, values * values, 0.0), axis=1)[:, None]
    mean = sum_values / C
    variance = sum_squares / C - mean * mean
    variance = tl.maximum(variance, 0.0)
    centered = values - mean
    invstd = libdevice.rsqrt(variance + 1.0e-6)

    row_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row_ids < TOTAL_ROWS
    tl.store(mean_ptr + row_ids[:, None], mean, mask=row_mask[:, None])
    tl.store(rsqrt_ptr + row_ids[:, None], invstd, mask=row_mask[:, None])

    channel_ids = tl.arange(0, BLOCK_C)
    channel_mask = channel_ids < C
    weight = tl.load(weight_ptr + channel_ids, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    bias = tl.load(bias_ptr + channel_ids, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    output = centered * invstd * weight + bias
    tl.store(out_ptr + offsets, output, mask=mask)


@oracle_impl(hardware="B200", point="3162d0ee", BLOCK_C=1024, BLOCK_ROWS=4, num_warps=4)
@oracle_impl(hardware="B200", point="366664ec", BLOCK_C=512, BLOCK_ROWS=16, num_warps=4)
@oracle_impl(hardware="B200", point="bf60dc4e", BLOCK_C=256, BLOCK_ROWS=16, num_warps=4)
@oracle_impl(hardware="B200", point="32d9a8b7", BLOCK_C=128, BLOCK_ROWS=32, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_ROWS: int, num_warps: int):
    arg0_1, arg1_1, arg2_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    total_rows = n * h * w

    mean = torch.empty_strided(
        (n, h, w, 1),
        (h * w, w, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (n, h, w, 1),
        (h * w, w, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, c * w, c),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _channel_layernorm_stats_bf16_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        mean,
        rsqrt,
        out,
        C=c,
        TOTAL_ROWS=total_rows,
        BLOCK_C=BLOCK_C,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=3,
    )
    return mean, rsqrt, out
