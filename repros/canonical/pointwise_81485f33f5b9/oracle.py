"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 batch-norm inference residual ReLU tail by hoisting only the channel-invariant reciprocal-sqrt term and then applying the exact captured pointwise order, bf16 cast, bf16 residual add, and NaN-preserving ReLU in one NCHW-tiled output kernel, whereas Inductor currently keeps the broadcast normalization and residual epilogue in a generic elementwise schedule for every output element; Inductor cannot do this today because its scheduler/codegen does not form a channel-parameter producer plus full-scope tiled epilogue for this BN-affine/residual/ReLU pattern while preserving the explicit bf16 rounding boundary; the fix is SCHEDULER_FUSION: add a broadcast-invariant BN parameter producer and fused residual-ReLU epilogue schedule for dense NCHW pointwise tails."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sqrt_rn(a):
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rcp_rn(a):
    return tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _invstd_kernel(
    var_ptr,
    invstd_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offsets < C
    var = tl.load(var_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    eps = tl.full((BLOCK_C,), 1.0e-5, tl.float32)
    one = tl.full((BLOCK_C,), 1.0, tl.float32)
    invstd = _rcp_rn(_sqrt_rn(_add_rn(var, eps)))
    invstd = _mul_rn(invstd, one)
    tl.store(invstd_ptr + offsets, invstd, mask=mask)


@triton.jit
def _bn_residual_relu_kernel(
    mean_ptr,
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    hw_block = tl.program_id(2)

    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_mask = c_offsets < C
    hw_mask = hw_offsets < HW
    mask = c_mask[:, None] & hw_mask[None, :]
    offsets = n * C * HW + c_offsets[:, None] * HW + hw_offsets[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    centered = _sub_rn(x, mean[:, None])
    normalized = _mul_rn(centered, invstd[:, None])
    scaled = _mul_rn(normalized, weight[:, None])
    biased = _add_rn(scaled, bias[:, None])
    affine_bf16 = biased.to(tl.bfloat16)
    residual_sum = _add_rn(affine_bf16.to(tl.float32), residual).to(tl.bfloat16)
    residual_sum_f32 = residual_sum.to(tl.float32)
    relu = tl.where(
        residual_sum_f32 != residual_sum_f32,
        residual_sum_f32,
        tl.maximum(residual_sum_f32, 0.0),
    )
    tl.store(out_ptr + offsets, relu.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="85db042c", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="0af2d571", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="3aeeed95", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="fd6e1886", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="cd893d5b", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="28e6bebb", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="08581e62", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="50714806", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="4c0cf88f", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="6b748dff", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="7111f2e4", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2e6c4615", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e626f5a9", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="40170966", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e2161da4", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="72f4709b", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="cce8fbb3", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2f5dac5e", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="3bac8de4", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="592ca691", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e20a55a6", BLOCK_C=16, BLOCK_HW=64, BLOCK_INV=1024, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_HW: int,
    BLOCK_INV: int,
    num_warps: int,
):
    mean, x, var, weight, bias, residual = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width

    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    invstd = torch.empty((channels,), device=x.device, dtype=torch.float32)

    _invstd_kernel[(triton.cdiv(channels, BLOCK_INV),)](
        var,
        invstd,
        C=channels,
        BLOCK_C=BLOCK_INV,
        num_warps=4,
    )
    _bn_residual_relu_kernel[
        (batch, triton.cdiv(channels, BLOCK_C), triton.cdiv(hw, BLOCK_HW))
    ](
        mean,
        x,
        weight,
        bias,
        residual,
        invstd,
        out,
        C=channels,
        HW=hw,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
    )
    return out
