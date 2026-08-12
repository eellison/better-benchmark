"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full three-branch RepVGG BN-inference affine sum/ReLU for every returned channels-last bf16 element, materializing each branch's bf16 cast plus both bf16 add boundaries; whereas Inductor schedules the decomposed broadcast normalization and adds as a generic pointwise fusion over expanded elements; Inductor cannot do this today because its pointwise scheduler lacks a three-branch BN-affine epilogue template that hoists channel-only normalization while preserving bf16 rounding and NaN-preserving ReLU semantics; the fix is NEW_PATTERN: add a guarded RepVGG triple BN-inference add/ReLU template with channel-tiled scalar reuse and explicit bf16 cast/add boundaries."""

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
def _triple_bn_nhwc_kernel(
    mean0_ptr,
    x0_ptr,
    var0_ptr,
    weight0_ptr,
    bias0_ptr,
    mean1_ptr,
    x1_ptr,
    var1_ptr,
    weight1_ptr,
    bias1_ptr,
    mean2_ptr,
    x2_ptr,
    var2_ptr,
    weight2_ptr,
    bias2_ptr,
    branch0_ptr,
    branch1_ptr,
    branch2_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S_N: tl.constexpr,
    S_C: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cidx = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = nhw // (H * W)
    rem = nhw - n * (H * W)
    h = rem // W
    w = rem - h * W
    mask = (nhw[:, None] < (N * H * W)) & (cidx[None, :] < C)
    offsets = n[:, None] * S_N + cidx[None, :] * S_C + h[:, None] * S_H + w[:, None] * S_W

    mean0 = tl.load(mean0_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    var0 = tl.load(var0_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    var1 = tl.load(var1_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    var2 = tl.load(var2_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    inv0 = _f32_mul(_f32_div(1.0, _f32_sqrt(_f32_add(var0, EPS_VALUE))), 1.0)
    inv1 = _f32_mul(_f32_div(1.0, _f32_sqrt(_f32_add(var1, EPS_VALUE))), 1.0)
    inv2 = _f32_mul(_f32_div(1.0, _f32_sqrt(_f32_add(var2, EPS_VALUE))), 1.0)

    branch0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0[None, :]), inv0[None, :]), weight0[None, :]), bias0[None, :])
    branch1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1[None, :]), inv1[None, :]), weight1[None, :]), bias1[None, :])
    branch2 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x2, mean2[None, :]), inv2[None, :]), weight2[None, :]), bias2[None, :])

    tl.store(branch0_ptr + offsets, branch0.to(tl.bfloat16), mask=mask)
    tl.store(branch1_ptr + offsets, branch1.to(tl.bfloat16), mask=mask)
    tl.store(branch2_ptr + offsets, branch2.to(tl.bfloat16), mask=mask)


@triton.jit
def _bf16_add3_relu_kernel(
    branch0_ptr,
    branch1_ptr,
    branch2_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    branch0 = tl.load(branch0_ptr + offsets, mask=mask, other=0.0)
    branch1 = tl.load(branch1_ptr + offsets, mask=mask, other=0.0)
    branch2 = tl.load(branch2_ptr + offsets, mask=mask, other=0.0)
    add01 = (branch0 + branch1).to(tl.bfloat16)
    add012 = (add01 + branch2).to(tl.bfloat16)
    relu = tl.where(add012 != add012, add012, tl.maximum(add012, 0.0))
    tl.store(out_ptr + offsets, relu, mask=mask)


def _launch(inputs, *, BLOCK_M, BLOCK_C, num_warps):
    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
    ) = inputs
    n, c, h, w = x0.shape
    s_n, s_c, s_h, s_w = x0.stride()
    branch0 = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    branch1 = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    branch2 = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    out = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    _triple_bn_nhwc_kernel[(triton.cdiv(n * h * w, BLOCK_M), triton.cdiv(c, BLOCK_C))](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        branch0,
        branch1,
        branch2,
        N=n,
        C=c,
        H=h,
        W=w,
        S_N=s_n,
        S_C=s_c,
        S_H=s_h,
        S_W=s_w,
        EPS_VALUE=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _bf16_add3_relu_kernel[(triton.cdiv(x0.numel(), BLOCK_M * BLOCK_C),)](
        branch0,
        branch1,
        branch2,
        out,
        TOTAL=x0.numel(),
        BLOCK=BLOCK_M * BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="319d51bf", BLOCK_M=8, BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="8f8763aa", BLOCK_M=8, BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="c9e29114", BLOCK_M=8, BLOCK_C=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_C, num_warps):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, num_warps=num_warps)
