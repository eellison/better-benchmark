"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DCGAN bf16 BatchNorm-inference affine plus leaky-ReLU pointwise scope with a channel/spatial Triton tile, including bf16-to-fp32 stat conversion, `sqrt(var + 1e-5)` then reciprocal, the fp32 normalization and affine multiply/add order, the observable bf16 materialization before the `gt(0)` leaky-ReLU branch, and the final bf16 output store, whereas Inductor lowers the same full expression through generic pointwise scheduling; Inductor cannot remove a producer, reduction, scatter, alias, or output because this full scope is dominated by mandatory activation reads, channel-stat broadcasts, and output writes, with only limited channel-invariant hoisting available; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise case unless broader pointwise/broadcast codegen changes move both paths."""

import torch
import triton
import triton.language as tl

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
def _f32_sqrt(a):
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_affine_leaky_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    channel_mask = channels < CHANNELS
    hw_mask = hw < HW
    mask = channel_mask[:, None] & hw_mask[None, :]
    offsets = batch * CHANNELS * HW + channels[:, None] * HW + hw[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[:, None])
    var_eps = _f32_add(var, 1.0e-5)
    invstd = _f32_div(1.0, _f32_sqrt(var_eps))
    invstd = _f32_mul(invstd, 1.0)
    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, weight[:, None])
    affine = _f32_add(scaled, bias[:, None])
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    negative = _f32_mul(rounded, 0.2)
    out = tl.where(rounded > 0.0, rounded, negative)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def _launch(inputs, *, CHANNELS, HW, BLOCK_C, BLOCK_HW, num_warps, num_stages):
    mean, x, var, weight, bias = inputs
    batch = int(x.shape[0])
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    grid = (batch, triton.cdiv(CHANNELS, BLOCK_C), triton.cdiv(HW, BLOCK_HW))
    _bn_affine_leaky_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        CHANNELS=CHANNELS,
        HW=HW,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# ade1975e: (T([512], bf16), T([256,512,4,4], bf16), T([512], bf16), ...)
@oracle_impl(hardware="B200", point="ade1975e", CHANNELS=512, HW=16, BLOCK_C=64, BLOCK_HW=16, num_warps=4, num_stages=3)
# 359745db: (T([256], bf16), T([256,256,8,8], bf16), T([256], bf16), ...)
@oracle_impl(hardware="B200", point="359745db", CHANNELS=256, HW=64, BLOCK_C=16, BLOCK_HW=64, num_warps=4, num_stages=3)
# 177337c2: (T([128], bf16), T([256,128,16,16], bf16), T([128], bf16), ...)
@oracle_impl(hardware="B200", point="177337c2", CHANNELS=128, HW=256, BLOCK_C=4, BLOCK_HW=256, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    CHANNELS: int,
    HW: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        CHANNELS=CHANNELS,
        HW=HW,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
