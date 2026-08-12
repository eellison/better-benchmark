"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 UNet BN-inference affine, explicit bf16 cast, ReLU materialization, and 2x2 stride-2 low-memory maxpool value output by evaluating each non-overlapping pool window once, writing the returned ReLU tensor and pooled tensor from the same producer while a tail kernel writes the odd-width ReLU column, whereas Inductor materializes the full bf16 ReLU activation and then schedules the pooling stencil as separate generic work; Inductor cannot do this today because scheduler fusion does not sink a returned pointwise producer into low-memory maxpool while also preserving the producer as a visible output; the fix is SCHEDULER_FUSION: allow multi-output stencil fusion that writes the materialized producer and the dependent pool value in one loop nest with exact dtype boundaries."""

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
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_relu_value(x, mean, invstd, weight, bias):
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)
    rounded = _round_to_bf16_f32(affine)
    return tl.where((rounded > 0.0) | (rounded != rounded), rounded, 0.0)


@triton.jit
def _take_pool(candidate, candidate_offset, best, best_offset):
    take = (candidate > best) | (candidate != candidate)
    return tl.where(take, candidate, best), tl.where(take, candidate_offset, best_offset)


@triton.jit
def _bn_relu_pool2x2_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    relu_ptr,
    pool_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL

    ow = linear % OUT_W
    tmp = linear // OUT_W
    oh = tmp % OUT_H
    c = tmp // OUT_H

    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = _f32_mul(_f32_div(1.0, _f32_sqrt(_f32_add(var, 1.0e-5))), 1.0)

    in_h = oh * 2
    in_w = ow * 2
    base = c * H * W + in_h * W + in_w

    x00 = tl.load(x_ptr + base, mask=active, other=0.0).to(tl.float32)
    x01 = tl.load(x_ptr + base + 1, mask=active, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + base + W, mask=active, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + base + W + 1, mask=active, other=0.0).to(tl.float32)

    y00 = _bn_relu_value(x00, mean, invstd, weight, bias)
    y01 = _bn_relu_value(x01, mean, invstd, weight, bias)
    y10 = _bn_relu_value(x10, mean, invstd, weight, bias)
    y11 = _bn_relu_value(x11, mean, invstd, weight, bias)

    tl.store(relu_ptr + base, y00, mask=active)
    tl.store(relu_ptr + base + 1, y01, mask=active)
    tl.store(relu_ptr + base + W, y10, mask=active)
    tl.store(relu_ptr + base + W + 1, y11, mask=active)

    best = y00
    best_offset = tl.zeros((BLOCK,), dtype=tl.int32)
    best, best_offset = _take_pool(y01, 1, best, best_offset)
    best, best_offset = _take_pool(y10, 2, best, best_offset)
    best, best_offset = _take_pool(y11, 3, best, best_offset)
    tl.store(pool_ptr + linear, best, mask=active)


@triton.jit
def _bn_relu_last_col_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    relu_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL
    h = linear % H
    c = linear // H

    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = _f32_mul(_f32_div(1.0, _f32_sqrt(_f32_add(var, 1.0e-5))), 1.0)

    offset = c * H * W + h * W + (W - 1)
    x = tl.load(x_ptr + offset, mask=active, other=0.0).to(tl.float32)
    y = _bn_relu_value(x, mean, invstd, weight, bias)
    tl.store(relu_ptr + offset, y, mask=active)


def _launch(inputs, *, C, H, W, BLOCK, TAIL_BLOCK, num_warps):
    mean, x, var, weight, bias, _shape0, _shape1 = inputs
    out_h = H // 2
    out_w = W // 2
    relu = torch.empty_strided((1, C, H, W), (C * H * W, H * W, W, 1), device=x.device, dtype=torch.bfloat16)
    pool = torch.empty_strided((1, C, out_h, out_w), (C * out_h * out_w, out_h * out_w, out_w, 1), device=x.device, dtype=torch.bfloat16)
    total_pool = C * out_h * out_w

    _bn_relu_pool2x2_kernel[(triton.cdiv(total_pool, BLOCK),)](
        mean,
        x,
        var,
        weight,
        bias,
        relu,
        pool,
        C=C,
        H=H,
        W=W,
        OUT_H=out_h,
        OUT_W=out_w,
        TOTAL=total_pool,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _bn_relu_last_col_kernel[(triton.cdiv(C * H, TAIL_BLOCK),)](
        mean,
        x,
        var,
        weight,
        bias,
        relu,
        C=C,
        H=H,
        W=W,
        TOTAL=C * H,
        BLOCK=TAIL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return relu, pool


@oracle_impl(hardware="B200", point="3f17dc95", C=512, H=80, W=119, BLOCK=256, TAIL_BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="535abe76", C=256, H=160, W=239, BLOCK=256, TAIL_BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="0ce7c2f4", C=128, H=320, W=479, BLOCK=256, TAIL_BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="4f218228", C=64, H=640, W=959, BLOCK=256, TAIL_BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, C, H, W, BLOCK, TAIL_BLOCK, num_warps):
    return _launch(inputs, C=C, H=H, W=W, BLOCK=BLOCK, TAIL_BLOCK=TAIL_BLOCK, num_warps=num_warps)
