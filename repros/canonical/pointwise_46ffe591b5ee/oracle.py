"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet bf16 inference BatchNorm-affine, explicit bf16 SiLU boundary, and `[1,2,1,2]` constant-pad scope with channels-last Triton kernels that reuse each channel's mean/variance/weight/bias across spatial tiles and zero-fill only the pad fringe, whereas Inductor lowers the decomposed broadcast pointwise chain, natural-exp SiLU, bf16 casts, and padded output layout through generic pointwise/pad scheduling; Inductor cannot do this today because its scheduler/codegen does not fuse a channel-broadcast BN epilogue with an explicit bf16-rounded activation and a spatial pad stencil into one layout-aware plan; the fix is SCHEDULER_FUSION: teach pointwise scheduling to hoist per-channel BN parameters, preserve bf16 rounding boundaries, and sink fixed spatial padding into the final channels-last materialization."""

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
def _bn_silu_pad_interior_nhwc_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)[:, None]
    c_offsets = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]

    hw_mask = hw_offsets < (HEIGHT * WIDTH)
    c_mask = c_offsets < CHANNELS
    mask = hw_mask & c_mask

    h = hw_offsets // WIDTH
    w = hw_offsets - h * WIDTH

    input_offsets = n * CHANNELS * HEIGHT * WIDTH + hw_offsets * CHANNELS + c_offsets
    out_h: tl.constexpr = HEIGHT + 3
    out_w: tl.constexpr = WIDTH + 3
    output_offsets = n * CHANNELS * out_h * out_w + ((h + 1) * out_w + (w + 1)) * CHANNELS + c_offsets

    x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    var_eps = _f32_add(var, 0.001)
    sqrt = tl.sqrt_rn(var_eps)
    recip = _f32_div(1.0, sqrt)
    scale = _f32_mul(recip, 1.0)
    normalized = _f32_mul(centered, scale)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    rounded = affine.to(tl.bfloat16).to(tl.float32)
    exp_term = libdevice.exp(_f32_sub(0.0, rounded))
    denom = _f32_add(exp_term, 1.0)
    silu = _f32_div(rounded, denom)
    tl.store(out_ptr + output_offsets, silu.to(tl.bfloat16), mask=mask)


@triton.jit
def _zero_pad_1_2_1_2_nhwc_kernel(
    out_ptr,
    TOTAL_PAD: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL_PAD

    out_h: tl.constexpr = HEIGHT + 3
    out_w: tl.constexpr = WIDTH + 3
    pad_pixels: tl.constexpr = 3 * (HEIGHT + WIDTH + 3)

    c = offsets % CHANNELS
    tmp = offsets // CHANNELS
    pos = tmp % pad_pixels
    n = tmp // pad_pixels

    top = pos < out_w
    bottom = (pos >= out_w) & (pos < 3 * out_w)

    bottom_pos = pos - out_w
    bottom_h = HEIGHT + 1 + bottom_pos // out_w
    bottom_w = bottom_pos - (bottom_pos // out_w) * out_w

    side_pos = pos - 3 * out_w
    side_row = side_pos // 3
    side_slot = side_pos - side_row * 3
    side_h = side_row + 1
    side_w = tl.where(side_slot == 0, 0, WIDTH + side_slot)

    h = tl.where(top, 0, tl.where(bottom, bottom_h, side_h))
    w = tl.where(top, pos, tl.where(bottom, bottom_w, side_w))
    out_offsets = n * CHANNELS * out_h * out_w + (h * out_w + w) * CHANNELS + c
    tl.store(out_ptr + out_offsets, 0.0, mask=mask)


@oracle_impl(hardware="B200", point="8162a5bc", BLOCK_HW=32, BLOCK_C=64, PAD_BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="51719261", BLOCK_HW=64, BLOCK_C=64, PAD_BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int, PAD_BLOCK: int, num_warps: int):
    mean, x, var, weight, bias, _pad = inputs
    batch, channels, height, width = x.shape
    out_h = height + 3
    out_w = width + 3
    out = torch.empty_strided(
        (batch, channels, out_h, out_w),
        (channels * out_h * out_w, 1, out_w * channels, channels),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _bn_silu_pad_interior_nhwc_kernel[
        (batch, triton.cdiv(height * width, BLOCK_HW), triton.cdiv(channels, BLOCK_C))
    ](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )

    pad_pixels = 3 * (height + width + 3)
    total_pad = batch * channels * pad_pixels
    _zero_pad_1_2_1_2_nhwc_kernel[(triton.cdiv(total_pad, PAD_BLOCK),)](
        out,
        TOTAL_PAD=total_pad,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        BLOCK=PAD_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out
