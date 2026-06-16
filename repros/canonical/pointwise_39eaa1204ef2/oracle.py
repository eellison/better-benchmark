"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full VisFormer two-stage bf16 normalization epilogue, hoisting channelwise reciprocal-sqrt terms while preserving the first bf16 affine cast, residual bf16 add, second affine, final bf16 cast, channels-last strides, and both returned tensors; whereas Inductor lowers the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add graph as generic pointwise code over each element; Inductor cannot do this today because its algebraic simplifier/codegen does not canonicalize serial per-channel normalization epilogues with intermediate bf16 materialization and multi-output scope into shared channel-invariant terms; the fix is ALGEBRAIC_ELIMINATION: add a guarded rewrite that hoists channel invariants and emits the residual add plus dependent output in one full-scope pointwise kernel."""

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
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sqrt(x):
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_rcp(x):
    return tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _invstd_kernel(
    var1_ptr,
    var2_ptr,
    inv_ptr,
    C: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c < C
    var1 = tl.load(var1_ptr + c, mask=mask, other=1.0).to(tl.float32)
    var2 = tl.load(var2_ptr + c, mask=mask, other=1.0).to(tl.float32)
    inv1 = _f32_rcp(_f32_sqrt(_f32_add(var1, EPS_VALUE)))
    inv2 = _f32_rcp(_f32_sqrt(_f32_add(var2, EPS_VALUE)))
    tl.store(inv_ptr + c, inv1, mask=mask)
    tl.store(inv_ptr + C + c, inv2, mask=mask)


@triton.jit
def _dual_norm_kernel(
    mean1_ptr,
    x_ptr,
    inv_ptr,
    weight1_ptr,
    bias1_ptr,
    residual_ptr,
    mean2_ptr,
    weight2_ptr,
    bias2_ptr,
    out1_ptr,
    out2_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < TOTAL
    c = offsets % C
    residual_offsets = offsets % (C * HW)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    inv1 = tl.load(inv_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    inv2 = tl.load(inv_ptr + C + c, mask=mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    stage1 = _f32_mul(_f32_sub(x, mean1), inv1)
    stage1 = _f32_add(_f32_mul(stage1, weight1), bias1)
    stage1_bf16 = _round_bf16_to_f32(stage1)
    add_residual = _round_bf16_to_f32(_f32_add(stage1_bf16, residual))
    stage2 = _f32_mul(_f32_sub(add_residual, mean2), inv2)
    stage2 = _f32_add(_f32_mul(stage2, weight2), bias2)

    tl.store(out1_ptr + offsets, add_residual.to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + offsets, stage2.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="97d7b07b", BLOCK_N=1024, BLOCK_C=1024, num_warps=4)
@oracle_impl(hardware="B200", point="bd2d0775", BLOCK_N=1024, BLOCK_C=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f1b0313e", BLOCK_N=1024, BLOCK_C=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_C: int, num_warps: int):
    mean1, x, var1, weight1, bias1, residual, mean2, var2, weight2, bias2 = inputs
    batch, channels, height, width = x.shape
    total = x.numel()
    hw = height * width
    inv = torch.empty_strided((2, channels), (channels, 1), device=x.device, dtype=torch.float32)
    out1 = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    out2 = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    _invstd_kernel[(triton.cdiv(channels, BLOCK_C),)](
        var1,
        var2,
        inv,
        C=channels,
        EPS_VALUE=EPS,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    _dual_norm_kernel[(triton.cdiv(total, BLOCK_N),)](
        mean1,
        x,
        inv,
        weight1,
        bias1,
        residual,
        mean2,
        weight2,
        bias2,
        out1,
        out2,
        TOTAL=total,
        C=channels,
        HW=hw,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out1, out2
