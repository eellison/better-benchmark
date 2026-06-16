"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT bf16 BN-inference affine, explicit bf16 round, exact-exp SiLU, and residual channel-cat scope in one channels-last Triton storage kernel, whereas Inductor schedules the decomposed normalization/activation and cat materialization through generic pointwise regions; Inductor cannot do this today because its scheduler does not sink per-channel BN/SILU producers into a fixed channel-cat writer while preserving the bf16 rounding boundaries and final channels-last output layout; the fix is SCHEDULER_FUSION: add a guarded channel-cat pointwise fusion path that emits direct paired-segment stores with reusable channel parameters and exact activation codegen."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _bn_silu_cat_channels_last_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < TOTAL

    out_c = offsets % (2 * C)
    tmp = offsets // (2 * C)
    w = tmp % W
    tmp = tmp // W
    h = tmp % H
    n = tmp // H

    is_residual = out_c < C
    source_c = tl.where(is_residual, out_c, out_c - C)
    source_offsets = ((n * H + h) * W + w) * C + source_c

    residual = tl.load(residual_ptr + source_offsets, mask=mask & is_residual, other=0.0)

    x = tl.load(x_ptr + source_offsets, mask=mask & ~is_residual, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + source_c, mask=mask & ~is_residual, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + source_c, mask=mask & ~is_residual, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + source_c, mask=mask & ~is_residual, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + source_c, mask=mask & ~is_residual, other=0.0).to(tl.float32)

    sqrt = libdevice.sqrt(var + EPS_VALUE)
    invstd = (1.0 / sqrt) * 1.0
    affine = ((x - mean) * invstd) * weight + bias
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    silu = rounded / (libdevice.exp(-rounded) + 1.0)
    silu_bf16 = silu.to(tl.bfloat16)

    value = tl.where(is_residual, residual, silu_bf16)
    tl.store(out_ptr + offsets, value, mask=mask)


def _launch(inputs, *, BLOCK_N: int, num_warps: int):
    mean, x, var, weight, bias, residual = inputs
    batch, channels, height, width = x.shape
    out_channels = channels * 2
    out = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * height * width, 1, out_channels * width, out_channels),
        device=x.device,
        dtype=torch.bfloat16,
    )
    total = batch * out_channels * height * width
    _bn_silu_cat_channels_last_kernel[(triton.cdiv(total, BLOCK_N),)](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        out,
        TOTAL=total,
        C=channels,
        H=height,
        W=width,
        EPS_VALUE=EPS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out


@oracle_impl(hardware="B200", point="1ff5655c", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="b0704cab", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="91c4b41a", BLOCK_N=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    return _launch(inputs, BLOCK_N=BLOCK_N, num_warps=num_warps)
