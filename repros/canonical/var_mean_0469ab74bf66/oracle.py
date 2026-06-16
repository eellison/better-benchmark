"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 bf16 channel LayerNorm inference scope in one Triton row kernel, including the bf16 NCHW input add rounded before the fp32 NHWC channel reduction, `var_mean(..., dim=3, correction=0, keepdim=True)`, eps=1e-6 rsqrt, bf16 scale/bias affine in fp32, final bf16 rounding, and the direct channels-last NCHW output store, whereas Inductor lowers the captured add/permute/var_mean/affine/cast/permute graph through generic normalization and layout pointwise schedules; Inductor cannot do this today because its normalization scheduler does not fuse the pre-reduction bf16 add, channel statistics, affine epilogue, and final layout view into one channel-layernorm schedule for this NHWC-over-NCHW pattern; the fix is SCHEDULER_FUSION: add a guarded ConvNeXt channel-layernorm schedule that preserves the bf16 add boundary while emitting the statistics and rounded affine store directly for channels-last points."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _channel_layernorm_add_bf16_kernel(
    x0_ptr,
    x1_ptr,
    weight_ptr,
    bias_ptr,
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

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values = (x0 + x1).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    sum_values = tl.sum(tl.where(mask, values, 0.0), axis=1)[:, None]
    mean = sum_values / C
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / C
    invstd = libdevice.rsqrt(variance + 1.0e-6)

    channel_ids = tl.arange(0, BLOCK_C)
    channel_mask = channel_ids < C
    weight = tl.load(weight_ptr + channel_ids, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    bias = tl.load(bias_ptr + channel_ids, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    output = centered * invstd * weight + bias
    tl.store(out_ptr + offsets, output.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


def _channels_last_stride(batch: int, channels: int, height: int, width: int) -> tuple[int, int, int, int]:
    return (channels * height * width, 1, channels * width, channels)


@oracle_impl(hardware="B200", point="274067f9", BLOCK_C=512, BLOCK_ROWS=16, num_warps=4)
@oracle_impl(hardware="B200", point="f412d821", BLOCK_C=256, BLOCK_ROWS=16, num_warps=4)
@oracle_impl(hardware="B200", point="29afe565", BLOCK_C=128, BLOCK_ROWS=32, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_ROWS: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    total_rows = n * h * w
    out = torch.empty_strided(
        (n, c, h, w),
        _channels_last_stride(n, c, h, w),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _channel_layernorm_add_bf16_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        C=c,
        TOTAL_ROWS=total_rows,
        BLOCK_C=BLOCK_C,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
