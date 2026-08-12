"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet bf16 inference BatchNorm-affine, explicit bf16 SiLU round trips, and right/bottom zero `constant_pad_nd` scope in one channels-last Triton output-space kernel, whereas Inductor lowers the normalization/activation producer and padded materialization through generic pointwise and pad scheduling fragments; Inductor cannot do this today because its scheduler does not sink the visible bf16 activation rounding boundaries through the pad consumer while specializing the fixed channels-last layout; the fix is SCHEDULER_FUSION: teach pointwise/pad scheduling to fuse inference BN+SiLU producers into final padded channels-last materializations while preserving natural `exp`, eps, casts, and output stride."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_silu_right_bottom_pad_nhwc_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr,
    OUT_STRIDE_C: tl.constexpr,
    OUT_STRIDE_H: tl.constexpr,
    OUT_STRIDE_W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hw = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)[:, None]
    c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]

    out_height: tl.constexpr = HEIGHT + 1
    out_width: tl.constexpr = WIDTH + 1
    out_hw: tl.constexpr = out_height * out_width

    oh = hw // out_width
    ow = hw - oh * out_width
    c_mask = c < CHANNELS
    valid = (hw < out_hw) & (oh < HEIGHT) & (ow < WIDTH) & c_mask
    store_mask = (hw < out_hw) & c_mask

    x_offsets = (
        n * X_STRIDE_N
        + c * X_STRIDE_C
        + oh * X_STRIDE_H
        + ow * X_STRIDE_W
    )
    out_offsets = (
        n * OUT_STRIDE_N
        + c * OUT_STRIDE_C
        + oh * OUT_STRIDE_H
        + ow * OUT_STRIDE_W
    )

    x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    var_eps = _f32_add(var, 0.001)
    std = tl.sqrt_rn(var_eps)
    reciprocal = _f32_div(1.0, std)
    scale = _f32_mul(reciprocal, 1.0)
    normalized = _f32_mul(centered, scale)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    rounded_affine = affine.to(tl.bfloat16)
    silu_input = rounded_affine.to(tl.float32)
    denom = _f32_add(libdevice.exp(-silu_input), 1.0)
    silu = _f32_div(silu_input, denom).to(tl.bfloat16)
    zero = tl.full((BLOCK_HW, BLOCK_C), 0.0, tl.float32).to(tl.bfloat16)
    out_value = tl.where(valid, silu, zero)

    tl.store(out_ptr + out_offsets, out_value, mask=store_mask)


def _launch(inputs, *, BLOCK_HW: int, BLOCK_C: int, num_warps: int):
    mean, x, var, weight, bias = inputs
    batch, channels, height, width = x.shape
    out_height = height + 1
    out_width = width + 1
    out = torch.empty_strided(
        (batch, channels, out_height, out_width),
        (channels * out_height * out_width, 1, out_width * channels, channels),
        device=x.device,
        dtype=torch.bfloat16,
    )

    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()
    grid = (
        batch,
        triton.cdiv(out_height * out_width, BLOCK_HW),
        triton.cdiv(channels, BLOCK_C),
    )
    _bn_silu_right_bottom_pad_nhwc_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        X_STRIDE_N=x_stride_n,
        X_STRIDE_C=x_stride_c,
        X_STRIDE_H=x_stride_h,
        X_STRIDE_W=x_stride_w,
        OUT_STRIDE_N=out_stride_n,
        OUT_STRIDE_C=out_stride_c,
        OUT_STRIDE_H=out_stride_h,
        OUT_STRIDE_W=out_stride_w,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=4,
    )
    return out


@oracle_impl(hardware="B200", point="9ab2d895", BLOCK_HW=32, BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="2e1844e5", BLOCK_HW=64, BLOCK_C=32, num_warps=4)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int, num_warps: int):
    return _launch(inputs, BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C, num_warps=num_warps)
