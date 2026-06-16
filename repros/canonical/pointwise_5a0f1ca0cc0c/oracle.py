"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet two-input channel-cat plus BN-inference affine/ReLU scope for all seven shape points by reading the fixed cat operands as a virtual channel layout, applying the captured fp32 sqrt/reciprocal affine math from bf16 parameters, preserving the explicit bf16 cast before NaN-preserving ReLU, and writing the final contiguous bf16 output directly, whereas Inductor materializes or generically indexes the channel cat before the downstream broadcast pointwise normalization; Inductor cannot do this today because its scheduler does not model fixed aten.cat producers as virtual multi-source layouts feeding channel-parameter pointwise consumers with exact dtype boundaries; the fix is SCHEDULER_FUSION: teach pointwise scheduling to sink static channel-cat source selection into BN-affine/ReLU codegen while preserving eps, casts, strides, and output scope."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5


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
def _relu_bf16_preserve_nan(x):
    zero = tl.full(x.shape, 0.0, tl.float32).to(tl.bfloat16)
    return tl.where((x > zero) | (x != x), x, zero)


@triton.jit
def _virtual_cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    batch = rows // CHANNELS
    channel = rows - batch * CHANNELS
    row_mask = rows < TOTAL_ROWS
    hw_mask = spatial < HW
    mask = row_mask[:, None] & hw_mask[None, :]

    in_x0 = channel < C0
    x0_offsets = (batch[:, None] * C0 + channel[:, None]) * HW + spatial[None, :]
    x1_channel = channel - C0
    x1_offsets = (batch[:, None] * C1 + x1_channel[:, None]) * HW + spatial[None, :]
    x0 = tl.load(x0_ptr + x0_offsets, mask=mask & in_x0[:, None], other=0.0).to(
        tl.float32
    )
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask & (~in_x0)[:, None], other=0.0).to(
        tl.float32
    )
    x = _f32_add(x0, x1)

    mean = tl.load(mean_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[:, None])
    var_eps = _f32_add(var, tl.full((BLOCK_ROWS,), 1.0e-5, tl.float32))
    invstd = _f32_div(tl.full((BLOCK_ROWS,), 1.0, tl.float32), tl.sqrt_rn(var_eps))
    invstd = _f32_mul(invstd, tl.full((BLOCK_ROWS,), 1.0, tl.float32))
    normed = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normed, weight[:, None])
    affine = _f32_add(scaled, bias[:, None])
    out = _relu_bf16_preserve_nan(
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    )

    out_offsets = rows[:, None] * HW + spatial[None, :]
    tl.store(out_ptr + out_offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="732e3257", BLOCK_ROWS=16, BLOCK_HW=64, num_warps=8)
@oracle_impl(hardware="B200", point="dca482b1", BLOCK_ROWS=8, BLOCK_HW=128, num_warps=8)
@oracle_impl(hardware="B200", point="a3bd7f06", BLOCK_ROWS=4, BLOCK_HW=256, num_warps=8)
@oracle_impl(hardware="B200", point="cfae6d5c", BLOCK_ROWS=2, BLOCK_HW=512, num_warps=8)
@oracle_impl(hardware="B200", point="8aa233bf", BLOCK_ROWS=8, BLOCK_HW=128, num_warps=8)
@oracle_impl(hardware="B200", point="1b91efd5", BLOCK_ROWS=8, BLOCK_HW=128, num_warps=8)
@oracle_impl(hardware="B200", point="320237ea", BLOCK_ROWS=4, BLOCK_HW=256, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_HW: int,
    num_warps: int,
):
    x0, x1, mean, var, weight, bias = inputs
    batch = int(x0.shape[0])
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    channels = c0 + c1
    hw = height * width
    out = torch.empty_strided(
        (batch, channels, height, width),
        (channels * hw, hw, width, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(batch * channels, BLOCK_ROWS), triton.cdiv(hw, BLOCK_HW))
    _virtual_cat_bn_relu_kernel[grid](
        x0,
        x1,
        mean,
        var,
        weight,
        bias,
        out,
        C0=c0,
        C1=c1,
        CHANNELS=channels,
        HW=hw,
        TOTAL_ROWS=batch * channels,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
