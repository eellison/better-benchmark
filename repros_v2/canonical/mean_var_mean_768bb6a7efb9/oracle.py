"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 residual add, bf16 9x9 spatial mean, 640-channel population var_mean LayerNorm, affine epilogue, final bf16 cast, and `[128,640]` output view in one Triton row kernel, whereas Inductor lowers the decomposed add/mean/as_strided/permute/var_mean/rsqrt/affine/cast/permute/view graph through generic normalization scheduling around a pooled intermediate; Inductor cannot do this today because its norm lowering does not fold a fixed spatial average producer with required bf16 rounding boundaries into the channel LayerNorm row plan; the fix is SCHEDULER_FUSION: add a guarded pooled-residual LayerNorm lowering that preserves casts, strides, and output scope while reducing the intermediate scheduling overhead."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-6


@triton.jit
def _round_to_bf16_f32(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _spatial_mean_kernel(
    x0_ptr,
    x1_ptr,
    pooled_ptr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c_offsets < 640
    acc = tl.zeros((BLOCK_C,), tl.float32)

    for h in tl.static_range(0, 9):
        for w in tl.static_range(0, 9):
            offsets = batch * 51840 + c_offsets + h * 5760 + w * 640
            x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            acc += _round_to_bf16_f32(x0 + x1)

    tl.store(pooled_ptr + batch * 640 + c_offsets, acc * (1.0 / 81.0), mask=mask)


@triton.jit
def _layernorm_kernel(
    pooled_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_C: tl.constexpr,
    EPS_: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < 640
    pooled = tl.load(pooled_ptr + row * 640 + cols, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.where(mask, pooled, 0.0)
    mean = tl.sum(pooled, axis=0) * (1.0 / 640.0)
    centered = pooled - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) * (1.0 / 640.0)
    invstd = tl.rsqrt(variance + EPS_)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (centered * invstd) * weight + bias
    tl.store(out_ptr + row * 640 + cols, out, mask=mask)


# 10a79381: two T([128,640,9,9], bf16, stride=(51840,1,5760,640)) inputs.
@oracle_impl(
    hardware="B200",
    point="10a79381",
    SPATIAL_BLOCK_C=512,
    LN_BLOCK_C=1024,
    spatial_num_warps=4,
    ln_num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    SPATIAL_BLOCK_C: int,
    LN_BLOCK_C: int,
    spatial_num_warps: int,
    ln_num_warps: int,
):
    x0, x1, weight, bias, shape0, stride0, shape1 = inputs
    pooled = torch.empty_strided(
        (int(shape1[0]), int(shape1[1])),
        (int(shape1[1]), 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        (int(shape1[0]), int(shape1[1])),
        (int(shape1[1]), 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    _spatial_mean_kernel[(128, triton.cdiv(640, SPATIAL_BLOCK_C))](
        x0,
        x1,
        pooled,
        BLOCK_C=SPATIAL_BLOCK_C,
        num_warps=spatial_num_warps,
        num_stages=3,
    )
    _layernorm_kernel[(128,)](
        pooled,
        weight,
        bias,
        out,
        BLOCK_C=LN_BLOCK_C,
        EPS_=EPS,
        num_warps=ln_num_warps,
        num_stages=3,
    )
    return out
