"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 channels-last BatchNorm-inference affine plus SiLU pointwise region in one channel/spatial tiled Triton kernel, including the sqrt/reciprocal normalization, the required bf16 rounding boundary before `x / (exp(-x) + 1)`, libdevice natural exp, final bf16 output, and channels-last output stride, whereas Inductor lowers the decomposed broadcast, cast, exp, div, and output cast graph as a generic fused pointwise kernel over flattened indexing; Inductor cannot do this today because the scheduler/codegen does not specialize per-channel broadcast pointwise fusion into a channels-last channel-by-spatial stencil that reuses channel parameters while preserving the visible bf16 cast boundary and activation numerics; the fix is SCHEDULER_FUSION: add a layout-aware BN-affine-plus-activation pointwise template for channels-last NCHW tensors with exact cast and libdevice exp boundaries."""

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
def _bn_silu_bf16_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_c = c_offsets < C
    mask_hw = hw_offsets < HW
    mask = mask_c[:, None] & mask_hw[None, :]

    offsets = n * C * HW + hw_offsets[None, :] * C + c_offsets[:, None]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)

    sqrt = tl.sqrt_rn(_f32_add(var, 1.0e-5))
    invstd = _f32_mul(_f32_div(1.0, sqrt), 1.0)
    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, weight[:, None])
    affine = _f32_add(scaled, bias[:, None])

    rounded = affine.to(tl.bfloat16).to(tl.float32)
    denom = _f32_add(libdevice.exp(-rounded), 1.0)
    out = _f32_div(rounded, denom).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


def _run(inputs, *, BLOCK_C: int, BLOCK_HW: int, num_warps: int, num_stages: int):
    mean, x, var, weight, bias = inputs
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    _bn_silu_bf16_kernel[(x.shape[0], triton.cdiv(c, BLOCK_C), triton.cdiv(hw, BLOCK_HW))](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        C=c,
        HW=hw,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


@oracle_impl(hardware="B200", point="d6d99242", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="ec18ed25", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="58353e1a", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="9f949812", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="9afa7692", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="a423f2c2", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="6129a191", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="39025e04", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="c4fb31ef", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="bf6f4282", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="c0e7b662", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="5e96e191", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="540fcc87", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="64799fb2", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="8162a5bc", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="d46c34b3", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="9ab2d895", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="51719261", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="9c97edfa", BLOCK_C=8, BLOCK_HW=64, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_HW: int, num_warps: int, num_stages: int):
    return _run(inputs, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW, num_warps=num_warps, num_stages=num_stages)
