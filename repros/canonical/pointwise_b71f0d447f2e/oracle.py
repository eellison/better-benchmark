"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Inception inference four-branch channels-last BN-affine/ReLU, returned static channel cat, and padded 3x3 avg_pool2d tail for all shape points, preserving Inductor's f32 sqrt/reciprocal affine math, explicit bf16 cast before NaN-preserving ReLU, channels-last output strides, and avg-pool accumulation order, whereas Inductor emits a generic fused cat pointwise kernel plus a separate generic avg_pool2d stencil over the materialized cat tensor; Inductor cannot do this today because scheduler fusion does not group sibling BN-affine branches with a static channel-cat layout and downstream pool consumer into a branch-specialized full-scope plan; the fix is SCHEDULER_FUSION: teach pointwise/stencil scheduling to sink static BN/ReLU branch producers into cat/pool materialization while preserving dtype boundaries and returned cat scope."""

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
def _relu_bf16_preserve_nan(x):
    zero = tl.full(x.shape, 0.0, tl.float32).to(tl.bfloat16)
    return tl.where((x > zero) | (x != x), x, zero)


@triton.jit
def _bn_relu_cat_branch_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,
    OUT_OFFSET: tl.constexpr,
    C: tl.constexpr,
    TOTAL_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channel = offsets % C
    tmp = offsets // C
    w_idx = tmp % W
    tmp = tmp // W
    h_idx = tmp % H
    n_idx = tmp // H

    x_offsets = n_idx * SN + channel * SC + h_idx * SH + w_idx * SW
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    var_eps = _f32_add(var, tl.full((BLOCK,), 0.001, tl.float32))
    inv = _f32_div(tl.full((BLOCK,), 1.0, tl.float32), tl.sqrt_rn(var_eps))
    inv = _f32_mul(inv, tl.full((BLOCK,), 1.0, tl.float32))
    affine = _f32_add(_f32_mul(_f32_mul(centered, inv), weight), bias)
    relu = _relu_bf16_preserve_nan(
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    )

    store_offsets = ((n_idx * H + h_idx) * W + w_idx) * TOTAL_C + OUT_OFFSET + channel
    tl.store(cat_ptr + store_offsets, relu, mask=mask)


@triton.jit
def _avg_pool3x3_kernel(
    cat_ptr,
    pool_ptr,
    TOTAL_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    out_c = offsets % TOTAL_C
    tmp = offsets // TOTAL_C
    ow = tmp % W
    tmp = tmp // W
    oh = tmp % H
    _ = out_c

    row = W * TOTAL_C
    left = ow > 0
    right = ow + 1 < W
    top = oh > 0
    bottom = oh + 1 < H

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    v00 = tl.load(cat_ptr + offsets - row - TOTAL_C, mask=mask & top & left, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v00)
    v01 = tl.load(cat_ptr + offsets - row, mask=mask & top, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v01)
    v02 = tl.load(cat_ptr + offsets - row + TOTAL_C, mask=mask & top & right, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v02)
    v10 = tl.load(cat_ptr + offsets - TOTAL_C, mask=mask & left, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v10)
    v11 = tl.load(cat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v11)
    v12 = tl.load(cat_ptr + offsets + TOTAL_C, mask=mask & right, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v12)
    v20 = tl.load(cat_ptr + offsets + row - TOTAL_C, mask=mask & bottom & left, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v20)
    v21 = tl.load(cat_ptr + offsets + row, mask=mask & bottom, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v21)
    v22 = tl.load(cat_ptr + offsets + row + TOTAL_C, mask=mask & bottom & right, other=0.0).to(tl.float32)
    acc = _f32_add(acc, v22)

    pooled = _f32_div(acc, tl.full((BLOCK,), 9.0, tl.float32))
    tl.store(
        pool_ptr + offsets,
        pooled.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _run(inputs, *, CAT_BLOCK: int, POOL_BLOCK: int, cat_warps: int, pool_warps: int):
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
        mean3,
        x3,
        var3,
        weight3,
        bias3,
    ) = inputs
    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    c2 = int(x2.shape[1])
    c3 = int(x3.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    total_c = c0 + c1 + c2 + c3
    total = n * total_c * h * w

    cat = torch.empty_strided(
        (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    pool = torch.empty_strided(
        (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    _bn_relu_cat_branch_kernel[(triton.cdiv(n * c0 * h * w, CAT_BLOCK),)](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        cat,
        OUT_OFFSET=0,
        C=c0,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=n * c0 * h * w,
        SN=int(x0.stride(0)),
        SC=int(x0.stride(1)),
        SH=int(x0.stride(2)),
        SW=int(x0.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=cat_warps,
        num_stages=3,
    )
    _bn_relu_cat_branch_kernel[(triton.cdiv(n * c1 * h * w, CAT_BLOCK),)](
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        cat,
        OUT_OFFSET=c0,
        C=c1,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=n * c1 * h * w,
        SN=int(x1.stride(0)),
        SC=int(x1.stride(1)),
        SH=int(x1.stride(2)),
        SW=int(x1.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=cat_warps,
        num_stages=3,
    )
    _bn_relu_cat_branch_kernel[(triton.cdiv(n * c2 * h * w, CAT_BLOCK),)](
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        cat,
        OUT_OFFSET=c0 + c1,
        C=c2,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=n * c2 * h * w,
        SN=int(x2.stride(0)),
        SC=int(x2.stride(1)),
        SH=int(x2.stride(2)),
        SW=int(x2.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=cat_warps,
        num_stages=3,
    )
    _bn_relu_cat_branch_kernel[(triton.cdiv(n * c3 * h * w, CAT_BLOCK),)](
        mean3,
        x3,
        var3,
        weight3,
        bias3,
        cat,
        OUT_OFFSET=c0 + c1 + c2,
        C=c3,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=n * c3 * h * w,
        SN=int(x3.stride(0)),
        SC=int(x3.stride(1)),
        SH=int(x3.stride(2)),
        SW=int(x3.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=cat_warps,
        num_stages=3,
    )
    _avg_pool3x3_kernel[(triton.cdiv(total, POOL_BLOCK),)](
        cat,
        pool,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=total,
        BLOCK=POOL_BLOCK,
        num_warps=pool_warps,
        num_stages=3,
    )
    return cat, pool


# 25fb017b: four 192-channel channels-last Inception branches, H=W=17.
@oracle_impl(hardware="B200", point="25fb017b", CAT_BLOCK=1024, POOL_BLOCK=512, cat_warps=4, pool_warps=4)
# 78045192: 64/64/96/64 channels-last Inception branches, H=W=35.
@oracle_impl(hardware="B200", point="78045192", CAT_BLOCK=1024, POOL_BLOCK=256, cat_warps=4, pool_warps=4)
# fd10b4e8: 64/64/96/32 channels-last Inception branches, H=W=35.
@oracle_impl(hardware="B200", point="fd10b4e8", CAT_BLOCK=1024, POOL_BLOCK=256, cat_warps=4, pool_warps=4)
def oracle_forward(
    inputs,
    *,
    CAT_BLOCK: int,
    POOL_BLOCK: int,
    cat_warps: int,
    pool_warps: int,
):
    return _run(
        inputs,
        CAT_BLOCK=CAT_BLOCK,
        POOL_BLOCK=POOL_BLOCK,
        cat_warps=cat_warps,
        pool_warps=pool_warps,
    )
