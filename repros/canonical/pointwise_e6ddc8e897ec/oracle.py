"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 BatchNorm-inference affine plus residual-add epilogue for all captured channels-last and contiguous NCHW points, preserving the fp32 mean/variance promotion, `sqrt(var + 1e-5)` then reciprocal order, fp32 affine math, explicit bf16 rounding before the residual add, final bf16 output dtype, and input-matching output strides, whereas Inductor lowers the same graph as a generic broadcasted pointwise kernel that recomputes channel-invariant sqrt/reciprocal work for every output element; Inductor cannot do this today because its pointwise scheduler does not hoist broadcast-invariant BN parameters into a per-channel producer while preserving the visible bf16 cast boundary and residual-add scope; the fix is ALGEBRAIC_ELIMINATION: canonicalize BN-inference affine residual epilogues so invariant per-channel coefficient work is computed once and reused by the output writer without changing math, dtype boundaries, or strides."""

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
def _invstd_kernel(var_ptr, invstd_ptr, C: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < C
    var = tl.load(var_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    sqrt = _sqrt_rn(_f32_add(var, 1.0e-5))
    invstd = _f32_mul(_rcp_rn(sqrt), 1.0)
    tl.store(invstd_ptr + offsets, invstd, mask=mask)


@triton.jit
def _bn_affine_residual_kernel(
    mean_ptr,
    x_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    if CHANNELS_LAST:
        c = offsets % C
    else:
        c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = _f32_sub(x, mean)
    y = _f32_mul(y, invstd)
    y = _f32_mul(y, weight)
    y = _f32_add(y, bias)
    rounded = y.to(tl.bfloat16)
    out = _f32_add(rounded.to(tl.float32), residual).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="1b9786b5", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="cbfa3b6c", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="dee7c6c7", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="2cdd53e8", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="02ba49e6", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="059bf3db", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="342a6237", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="c8441aee", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="fc6552ee", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="50e4898b", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="3bc177c9", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="a2ab0ad0", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="8858d870", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="cbb247bc", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="6f6fdd1f", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="661bb0c6", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="9147e0bf", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="6834e84b", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="4273c1f9", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="127b8ac2", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="0f6a0e52", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="cce36204", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="e62d3519", BLOCK=1024, COEFF_BLOCK=256)
@oracle_impl(hardware="B200", point="7638df56", BLOCK=1024, COEFF_BLOCK=256)
def oracle_forward(inputs, *, BLOCK, COEFF_BLOCK):
    mean, x, var, weight, bias, residual = inputs
    n, c, h, w = x.shape
    total = n * c * h * w
    hw = h * w
    channels_last = x.stride(1) == 1

    invstd = torch.empty((c,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    _invstd_kernel[(1,)](
        var,
        invstd,
        C=c,
        BLOCK=COEFF_BLOCK,
        num_warps=8,
    )
    _bn_affine_residual_kernel[(triton.cdiv(total, BLOCK),)](
        mean,
        x,
        invstd,
        weight,
        bias,
        residual,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        CHANNELS_LAST=channels_last,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out
