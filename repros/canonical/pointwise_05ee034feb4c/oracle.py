"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete pytorch_unet bf16 BN-inference affine, explicit bf16 rounding, NaN-preserving ReLU, fixed bilinear upsample, right pad, and final bf16 channel cat in one output kernel after a tiny invstd precompute, whereas Inductor lowers the BN/ReLU producer, four indexed bilinear gathers, pad, and cat through generic pointwise/layout scheduling around intermediate tensors; Inductor cannot do this today because its scheduler does not sink a broadcast BN-affine producer through fixed bilinear gather math into the padded cat store while preserving the bf16 cast boundary before ReLU; the fix is SCHEDULER_FUSION: recognize this fixed upsample-pad-cat consumer chain and emit direct final-layout stores from the fused producer."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SCALE_H = 0.4968553459119497
SCALE_W = 0.4978902953586498


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
def _invstd_kernel(var_ptr, invstd_ptr, C: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < C
    var = tl.load(var_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    sqrt_var = libdevice.sqrt(_f32_add(var, 1.0e-5))
    invstd = _f32_mul(1.0 / sqrt_var, 1.0)
    tl.store(invstd_ptr + offsets, invstd, mask=mask)


@triton.jit
def _relu_value(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    channels,
    h_idx,
    w_idx,
    H: tl.constexpr,
    W: tl.constexpr,
    mask,
):
    offsets = channels * H * W + h_idx * W + w_idx
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    zero = tl.full(rounded.shape, 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where(rounded != rounded, rounded, tl.maximum(rounded, zero))
    return relu.to(tl.float32)


@triton.jit
def _cat_upsample_kernel(
    x_ptr,
    skip_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    TOTAL: tl.constexpr,
    SCALE_H_C: tl.constexpr,
    SCALE_W_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    w_out = offsets % OUT_W
    h_out = (offsets // OUT_W) % OUT_H
    c_out = (offsets // (OUT_H * OUT_W)) % (C * 2)
    first_half = c_out < C
    c = tl.where(first_half, c_out, c_out - C)

    skip_offsets = c * OUT_H * OUT_W + h_out * OUT_W + w_out
    skip_value = tl.load(skip_ptr + skip_offsets, mask=mask & first_half, other=0.0)

    valid_upsample = mask & ~first_half & (w_out < (OUT_W - 1))
    y_float = _f32_mul(h_out.to(tl.float32), SCALE_H_C)
    y0 = y_float.to(tl.int64)
    y1 = tl.minimum(y0 + 1, IN_H - 1)
    y_weight = tl.minimum(tl.maximum(_f32_sub(y_float, y0.to(tl.float32)), 0.0), 1.0)

    x_float = _f32_mul(w_out.to(tl.float32), SCALE_W_C)
    x0 = x_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, IN_W - 1)
    x_weight = tl.minimum(tl.maximum(_f32_sub(x_float, x0.to(tl.float32)), 0.0), 1.0)

    v00 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y0, x0, IN_H, IN_W, valid_upsample)
    v01 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y0, x1, IN_H, IN_W, valid_upsample)
    v10 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y1, x0, IN_H, IN_W, valid_upsample)
    v11 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y1, x1, IN_H, IN_W, valid_upsample)

    row0 = _f32_add(v00, _f32_mul(_f32_sub(v01, v00), x_weight))
    row1 = _f32_add(v10, _f32_mul(_f32_sub(v11, v10), x_weight))
    interp = _f32_add(row0, _f32_mul(_f32_sub(row1, row0), y_weight))
    zero_bf16 = tl.full((BLOCK,), 0.0, tl.float32).to(tl.bfloat16)
    upsample_value = tl.where(w_out < (OUT_W - 1), interp.to(tl.bfloat16, fp_downcast_rounding="rtne"), zero_bf16)
    value = tl.where(first_half, skip_value, upsample_value)
    tl.store(out_ptr + offsets, value, mask=mask)


@oracle_impl(hardware="B200", point="d44a5eff", OUT_BLOCK=512, OUT_WARPS=4)
def oracle_forward(inputs, *, OUT_BLOCK: int, OUT_WARPS: int):
    mean, x, var, weight, bias, skip, _shape = inputs
    out = torch.empty_strided(
        (1, 512, 160, 239),
        (512 * 160 * 239, 160 * 239, 239, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    invstd = torch.empty_strided((256,), (1,), device=x.device, dtype=torch.float32)
    _invstd_kernel[(1,)](
        var,
        invstd,
        C=256,
        BLOCK=256,
        num_warps=4,
        num_stages=3,
    )
    _cat_upsample_kernel[(triton.cdiv(out.numel(), OUT_BLOCK),)](
        x,
        skip,
        mean,
        invstd,
        weight,
        bias,
        out,
        C=256,
        IN_H=80,
        IN_W=119,
        OUT_H=160,
        OUT_W=239,
        TOTAL=out.numel(),
        SCALE_H_C=SCALE_H,
        SCALE_W_C=SCALE_W,
        BLOCK=OUT_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=3,
    )
    return out
