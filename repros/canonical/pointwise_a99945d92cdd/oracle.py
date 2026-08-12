"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full GhostNet bf16 BN-inference branch, channel cat, and residual add in one Triton storage-order kernel, while preserving Inductor's decomposed sqrt->reciprocal arithmetic and the visible bf16 cat rounding boundary before the final residual add; Inductor lowers the captured graph as a cat-producing pointwise kernel followed by a second channels-last add kernel, so it pays an avoidable intermediate write/read and extra launch for this fixed channel-cat epilogue; the fix is SCHEDULER_FUSION: let fixed channel-concat producers feed the residual add consumer virtually without changing the fp32 math, casts, or output stride.
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_f32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


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
def _ghostnet_bn_cat_residual_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    residual_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    OUT_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    out_c = offsets % OUT_C
    hw = (offsets // OUT_C) % HW
    n = offsets // (OUT_C * HW)
    hi = out_c >= C
    c = out_c - tl.where(hi, C, 0)

    source_offsets = n * C * HW + hw * C + c
    skip = tl.load(skip_ptr + source_offsets, mask=mask & ~hi, other=0.0).to(
        tl.float32
    )

    x = tl.load(x_ptr + source_offsets, mask=mask & hi, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask & hi, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask & hi, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask & hi, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask & hi, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    denom = libdevice.sqrt(_f32_add(var, 1.0e-5))
    one = tl.full([BLOCK], 1.0, tl.float32)
    inv = _f32_div(one, denom)
    normalized = _f32_mul(centered, _f32_mul(inv, 1.0))
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    rounded_affine = tl.where(
        affine != affine,
        affine,
        _round_bf16_to_f32(affine),
    )
    cat_value = tl.where(hi, rounded_affine, skip)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, _f32_add(cat_value, residual), mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="08779877", BLOCK=256, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="833dc3e9", BLOCK=256, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="cc3ee47c", BLOCK=256, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="721d3683", BLOCK=256, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="84ed26eb", BLOCK=256, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="5a6caa49", BLOCK=256, num_warps=4, num_stages=4)
def oracle_forward(
    inputs,
    *,
    BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    mean, x, var, weight, bias, skip, residual = inputs
    out_shape = _shape_tuple(residual.shape)
    out_stride = _shape_tuple(residual.stride())
    c = int(mean.numel())
    hw = int(out_shape[2] * out_shape[3])
    total = int(residual.numel())

    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=residual.device,
        dtype=torch.bfloat16,
    )
    _ghostnet_bn_cat_residual_kernel[(triton.cdiv(total, BLOCK),)](
        mean,
        x,
        var,
        weight,
        bias,
        skip,
        residual,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        OUT_C=out_shape[1],
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
