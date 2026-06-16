"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete dual BN-inference affine add/ReLU epilogue for every returned element, preserving both fp32 `sqrt` then reciprocal normalization paths, each branch's bf16 cast boundary, the bf16 branch add, NaN-preserving ReLU, and the captured output stride; whereas Inductor lowers the same scope as a generic pointwise expression over each expanded tensor element; Inductor cannot do this today because its pointwise scheduler lacks a dual-branch BN-affine epilogue template that hoists per-channel scalars across NCHW and channels-last layouts while preserving bf16 rounding boundaries; the fix is NEW_PATTERN: add a guarded dual BN-inference add/ReLU template with layout-aware channel tiling and explicit cast-boundary preservation."""

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
def _dual_bn_nchw_kernel(
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
    branch0_ptr,
    branch1_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S_N: tl.constexpr,
    S_C: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    row_mask = rows < (N * C)
    hw_mask = hw < (H * W)
    mask = row_mask[:, None] & hw_mask[None, :]

    n = rows // C
    cidx = rows - n * C
    h = hw // W
    w = hw - h * W
    offsets = n[:, None] * S_N + cidx[:, None] * S_C + h[None, :] * S_H + w[None, :] * S_W

    mean0 = tl.load(mean0_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)
    var0 = tl.load(var0_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)
    var1 = tl.load(var1_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + cidx, mask=row_mask, other=0.0).to(tl.float32)

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv0 = _f32_div(1.0, _f32_sqrt(_f32_add(var0, EPS_VALUE)))
    inv1 = _f32_div(1.0, _f32_sqrt(_f32_add(var1, EPS_VALUE)))

    branch0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0[:, None]), inv0[:, None]), weight0[:, None]), bias0[:, None])
    branch1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1[:, None]), inv1[:, None]), weight1[:, None]), bias1[:, None])
    branch0_bf16 = branch0.to(tl.bfloat16)
    branch1_bf16 = branch1.to(tl.bfloat16)
    tl.store(branch0_ptr + offsets, branch0_bf16, mask=mask)
    tl.store(branch1_ptr + offsets, branch1_bf16, mask=mask)


@triton.jit
def _dual_bn_nhwc_kernel(
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
    branch0_ptr,
    branch1_ptr,
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

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv0 = _f32_div(1.0, _f32_sqrt(_f32_add(var0, EPS_VALUE)))
    inv1 = _f32_div(1.0, _f32_sqrt(_f32_add(var1, EPS_VALUE)))

    branch0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0[None, :]), inv0[None, :]), weight0[None, :]), bias0[None, :])
    branch1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1[None, :]), inv1[None, :]), weight1[None, :]), bias1[None, :])
    branch0_bf16 = branch0.to(tl.bfloat16)
    branch1_bf16 = branch1.to(tl.bfloat16)
    tl.store(branch0_ptr + offsets, branch0_bf16, mask=mask)
    tl.store(branch1_ptr + offsets, branch1_bf16, mask=mask)


@triton.jit
def _bf16_add_relu_kernel(
    branch0_ptr,
    branch1_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    branch0 = tl.load(branch0_ptr + offsets, mask=mask, other=0.0)
    branch1 = tl.load(branch1_ptr + offsets, mask=mask, other=0.0)
    added = (branch0 + branch1).to(tl.bfloat16)
    relu = tl.where(added != added, added, tl.maximum(added, 0.0))
    tl.store(out_ptr + offsets, relu, mask=mask)


def _launch(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1 = inputs
    n, c, h, w = x0.shape
    s_n, s_c, s_h, s_w = x0.stride()
    branch0 = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    branch1 = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    out = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=x0.dtype)
    if s_c == 1:
        _dual_bn_nhwc_kernel[(triton.cdiv(n * h * w, BLOCK_M), triton.cdiv(c, BLOCK_N))](
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
            branch0,
            branch1,
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
            BLOCK_C=BLOCK_N,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        _dual_bn_nchw_kernel[(triton.cdiv(n * c, BLOCK_M), triton.cdiv(h * w, BLOCK_N))](
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
            branch0,
            branch1,
            N=n,
            C=c,
            H=h,
            W=w,
            S_N=s_n,
            S_C=s_c,
            S_H=s_h,
            S_W=s_w,
            EPS_VALUE=EPS,
            BLOCK_ROWS=BLOCK_M,
            BLOCK_HW=BLOCK_N,
            num_warps=num_warps,
            num_stages=3,
        )
    _bf16_add_relu_kernel[(triton.cdiv(x0.numel(), BLOCK_M * BLOCK_N),)](
        branch0,
        branch1,
        out,
        TOTAL=x0.numel(),
        BLOCK=BLOCK_M * BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="82978638", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="3f9335c1", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="b691104d", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="7ee473b0", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="f5c6beac", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="d58c5fd0", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="5c9d3965", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="2c26d0d1", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="e4b64a12", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="1fa31355", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="5e705d74", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="88658619", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="cf13ba44", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="27f9b106", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="cc596ad9", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="68f48f4e", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="eceeb412", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="76020c9d", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="0f0aebfd", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="e006e72d", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="83969398", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="b188d91d", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="795eaade", BLOCK_M=8, BLOCK_N=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps)
