"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete two-output Visformer residual-add plus BN-inference affine scope in one channels-last Triton kernel, preserving the bf16 residual-add output, the bf16 cast boundary feeding the fp32 normalization path, `sqrt` then reciprocal, final bf16 affine output, and captured output strides; whereas Inductor lowers the same decomposed broadcasts as a generic pointwise expression with separate returned materializations; Inductor cannot do this today because its pointwise scheduler lacks a layout-aware BN-affine template that reuses per-channel scalars while simultaneously materializing the residual output and preserving bf16 round boundaries; the fix is NEW_PATTERN: add a guarded channels-last residual-add plus BN-affine pointwise template with explicit returned-output and cast-boundary handling."""

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
def _residual_bn_affine_nhwc_kernel(
    x0_ptr,
    x1_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_add_ptr,
    out_affine_ptr,
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

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_bf16 = _f32_add(x0, x1).to(tl.bfloat16)
    tl.store(out_add_ptr + offsets, add_bf16, mask=mask)

    mean = tl.load(mean_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cidx, mask=cidx < C, other=0.0).to(tl.float32)

    inv = _f32_div(1.0, _f32_sqrt(_f32_add(var, EPS_VALUE)))
    normalized = _f32_mul(_f32_sub(add_bf16.to(tl.float32), mean[None, :]), inv[None, :])
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])
    tl.store(out_affine_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _launch(inputs, *, BLOCK_M, BLOCK_C, num_warps):
    x0, x1, mean, var, weight, bias = inputs
    n, c, h, w = x0.shape
    s_n, s_c, s_h, s_w = x0.stride()
    out_add = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    out_affine = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    _residual_bn_affine_nhwc_kernel[(triton.cdiv(n * h * w, BLOCK_M), triton.cdiv(c, BLOCK_C))](
        x0,
        x1,
        mean,
        var,
        weight,
        bias,
        out_add,
        out_affine,
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
    return out_add, out_affine


@oracle_impl(hardware="B200", point="45e1ce96", BLOCK_M=8, BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="bf2c0e1a", BLOCK_M=8, BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="881ee73c", BLOCK_M=8, BLOCK_C=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_C, num_warps):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, num_warps=num_warps)
