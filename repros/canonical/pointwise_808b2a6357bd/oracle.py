"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 BN-affine/ReLU plus channel-cat scope with a paired channels-last Triton kernel that copies the existing branch into the first output half while computing the normalized branch into the second half, whereas Inductor emits a generic storage-linear cat pointwise kernel for the same fused expression; Inductor cannot do this today because its scheduler/codegen does not model fixed channel-cat materialization as a paired destination layout operation with per-channel BN parameter reuse across both channel halves; the fix is SCHEDULER_FUSION: teach cat materialization to own paired channel-segment stores and sink pointwise producers with channel side inputs into that schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _paired_bn_relu_cat_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    first_ptr,
    out_ptr,
    TOTAL_PIXELS: tl.constexpr,
    CHANNELS: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pixels = tl.program_id(0) * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (pixels[:, None] < TOTAL_PIXELS) & (channels[None, :] < CHANNELS)

    src_offsets = pixels[:, None] * CHANNELS + channels[None, :]
    out_offsets = pixels[:, None] * (CHANNELS * 2) + channels[None, :]

    first = tl.load(first_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, first, mask=mask)

    x = tl.load(x_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    channel_mask = channels < CHANNELS
    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    centered = x - mean[None, :]
    sqrt = libdevice.sqrt(var + EPS_VALUE)
    invstd = (1.0 / sqrt) * 1.0
    normalized = centered * invstd[None, :]
    affine = (normalized * weight[None, :]) + bias[None, :]
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    relu = tl.where(rounded <= 0.0, 0.0, rounded)
    tl.store(out_ptr + out_offsets + CHANNELS, relu.to(tl.bfloat16), mask=mask)


def _launch(inputs, *, BLOCK_PIX: int, BLOCK_C: int, num_warps: int):
    mean, x, var, weight, bias, first = inputs
    n, c, h, w = x.shape
    out = torch.empty_strided(
        (n, c * 2, h, w),
        (c * 2 * h * w, 1, c * 2 * w, c * 2),
        device=x.device,
        dtype=torch.bfloat16,
    )
    total_pixels = n * h * w
    _paired_bn_relu_cat_kernel[
        (triton.cdiv(total_pixels, BLOCK_PIX), triton.cdiv(c, BLOCK_C))
    ](
        mean,
        x,
        var,
        weight,
        bias,
        first,
        out,
        TOTAL_PIXELS=total_pixels,
        CHANNELS=c,
        EPS_VALUE=EPS,
        BLOCK_PIX=BLOCK_PIX,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="5339df69", BLOCK_PIX=2, BLOCK_C=512, num_warps=8)
@oracle_impl(hardware="B200", point="25326742", BLOCK_PIX=2, BLOCK_C=512, num_warps=8)
@oracle_impl(hardware="B200", point="adc5aeaa", BLOCK_PIX=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="ced32754", BLOCK_PIX=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="d663f205", BLOCK_PIX=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="eb1a198a", BLOCK_PIX=16, BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="19e4d616", BLOCK_PIX=32, BLOCK_C=32, num_warps=4)
@oracle_impl(hardware="B200", point="395e6a92", BLOCK_PIX=128, BLOCK_C=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_PIX: int, BLOCK_C: int, num_warps: int):
    return _launch(inputs, BLOCK_PIX=BLOCK_PIX, BLOCK_C=BLOCK_C, num_warps=num_warps)
