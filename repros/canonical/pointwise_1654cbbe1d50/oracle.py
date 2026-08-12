"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 MobileNetV3 BatchNorm-inference affine plus hard-swish pointwise scope in one layout-aware Triton kernel, preserving the sqrt/reciprocal normalization, the explicit bf16 rounding boundary before hard-swish, NaN-propagating clamp semantics, final bf16 output, and either channels-last or contiguous NCHW output stride, whereas Inductor lowers the decomposed broadcast affine and hard-swish graph as a generic flattened pointwise kernel; Inductor cannot do this today because its algebraic simplifier and pointwise codegen do not expose a reusable BN-affine plus hard-swish template that can choose exact captured arithmetic or channel-hoisted coefficients under the numerics policy; the fix is ALGEBRAIC_ELIMINATION: add a guarded BN-inference affine canonicalization feeding hard-swish that preserves cast boundaries and selects a layout-aware channel-indexed loop."""

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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
    ],
    key=["TOTAL", "C", "HW", "LAYOUT"],
)
@triton.jit
def _bn_hardswish_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    if LAYOUT == 0:
        channels = (offsets // HW) % C
    else:
        channels = offsets % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    sqrt = tl.sqrt_rn(_f32_add(var, 1.0e-5))
    invstd = _f32_mul(_f32_div(1.0, sqrt), 1.0)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)
    rounded = affine.to(tl.bfloat16).to(tl.float32)

    shifted = _f32_add(rounded, 3.0)
    lower = tl.where(shifted < 0.0, 0.0, shifted)
    upper = tl.where(lower > 6.0, 6.0, lower)
    out = _f32_div(_f32_mul(rounded, upper), 6.0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


def _run(inputs):
    mean, x, var, weight, bias = inputs
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    layout = 0 if x.is_contiguous() else 1
    hw = x.shape[2] * x.shape[3]
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)
    _bn_hardswish_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        TOTAL=x.numel(),
        C=x.shape[1],
        HW=hw,
        LAYOUT=layout,
    )
    return out


@oracle_impl(hardware="B200", point="3e244c1d")
@oracle_impl(hardware="B200", point="c37163dc")
@oracle_impl(hardware="B200", point="86f01d63")
@oracle_impl(hardware="B200", point="cd2bec6a")
@oracle_impl(hardware="B200", point="0a7477f3")
@oracle_impl(hardware="B200", point="4c825568")
@oracle_impl(hardware="B200", point="1b088102")
@oracle_impl(hardware="B200", point="c7c6ed1f")
@oracle_impl(hardware="B200", point="2c1989e8")
@oracle_impl(hardware="B200", point="d9aaabff")
@oracle_impl(hardware="B200", point="6cc76740")
@oracle_impl(hardware="B200", point="54ff84f2")
@oracle_impl(hardware="B200", point="78d4e1ef")
@oracle_impl(hardware="B200", point="08df9f87")
@oracle_impl(hardware="B200", point="a952c87d")
@oracle_impl(hardware="B200", point="05028dd8")
def oracle_forward(inputs):
    return _run(inputs)
