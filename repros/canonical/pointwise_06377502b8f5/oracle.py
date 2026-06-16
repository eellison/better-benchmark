"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Inception low-memory maxpool value branch, two bf16 batchnorm-affine ReLU branches, fixed channel concat, and padded 3x3 avg_pool2d over the materialized bf16 cat output with shape-specialized Triton kernels for both captured points, whereas Inductor lowers the maxpool, branch pointwise producers, concat, and avgpool as separate generic scheduling regions; Inductor cannot do this today because scheduler/codegen does not keep static channel-cat operands virtual across mixed pooling and broadcast-affine producers while preserving the returned cat side output and the bf16 cast-before-ReLU boundary; the fix is SCHEDULER_FUSION: teach the scheduler to fuse fixed channel-cat branch producers into pooling consumers while still emitting the exact returned bf16 cat layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_preserve_nan_bf16(x):
    xb = x.to(tl.bfloat16).to(tl.float32)
    return tl.where((xb > 0.0) | (xb != xb), xb, 0.0)


@triton.jit
def _bn_relu(x, mean, var, weight, bias):
    inv = 1.0 / tl.sqrt(var + 0.001)
    return _relu_preserve_nan_bf16((x - mean) * inv * weight + bias)


@triton.jit
def _cat_kernel(
    pool_in_ptr,
    mean0_ptr,
    branch0_ptr,
    var0_ptr,
    weight0_ptr,
    bias0_ptr,
    mean1_ptr,
    branch1_ptr,
    var1_ptr,
    weight1_ptr,
    bias1_ptr,
    cat_ptr,
    TOTAL: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    CPOOL: tl.constexpr,
    OUTC: tl.constexpr,
    HIN: tl.constexpr,
    WIN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % OUTC
    w = (offsets // OUTC) % W
    h = (offsets // (OUTC * W)) % H
    n = offsets // (OUTC * W * H)

    in_branch0 = c < C0
    c0 = tl.where(in_branch0, c, 0)
    off0 = n * C0 * H * W + h * W * C0 + w * C0 + c0
    x0 = tl.load(branch0_ptr + off0, mask=active & in_branch0, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + c0, mask=active & in_branch0, other=0.0).to(tl.float32)
    var0 = tl.load(var0_ptr + c0, mask=active & in_branch0, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + c0, mask=active & in_branch0, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + c0, mask=active & in_branch0, other=0.0).to(tl.float32)
    y0 = _bn_relu(x0, mean0, var0, weight0, bias0)

    in_branch1 = (c >= C0) & (c < C0 + C1)
    c1 = tl.where(in_branch1, c - C0, 0)
    off1 = n * C1 * H * W + h * W * C1 + w * C1 + c1
    x1 = tl.load(branch1_ptr + off1, mask=active & in_branch1, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c1, mask=active & in_branch1, other=0.0).to(tl.float32)
    var1 = tl.load(var1_ptr + c1, mask=active & in_branch1, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c1, mask=active & in_branch1, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c1, mask=active & in_branch1, other=0.0).to(tl.float32)
    y1 = _bn_relu(x1, mean1, var1, weight1, bias1)

    in_pool = c >= C0 + C1
    cp = tl.where(in_pool, c - C0 - C1, 0)
    best = tl.full([BLOCK], -float("inf"), tl.float32)
    for kh in tl.static_range(3):
        ih = h * 2 + kh
        for kw in tl.static_range(3):
            iw = w * 2 + kw
            poff = n * CPOOL * HIN * WIN + ih * WIN * CPOOL + iw * CPOOL + cp
            value = tl.load(pool_in_ptr + poff, mask=active & in_pool, other=-float("inf")).to(tl.float32)
            take = (value > best) | (value != value)
            best = tl.where(take, value, best)

    out = tl.where(in_branch0, y0, tl.where(in_branch1, y1, best))
    tl.store(cat_ptr + offsets, out, mask=active)


@triton.jit
def _avgpool_kernel(
    cat_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    OUTC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % OUTC
    w = (offsets // OUTC) % W
    h = (offsets // (OUTC * W)) % H
    n = offsets // (OUTC * W * H)
    acc = tl.zeros([BLOCK], tl.float32)
    for kh in tl.static_range(3):
        ih = h + kh - 1
        h_ok = (ih >= 0) & (ih < H)
        for kw in tl.static_range(3):
            iw = w + kw - 1
            valid = active & h_ok & (iw >= 0) & (iw < W)
            src = n * OUTC * H * W + ih * W * OUTC + iw * OUTC + c
            acc += tl.load(cat_ptr + src, mask=valid, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, acc * 0.1111111111111111, mask=active)


@oracle_impl(hardware="B200", point="ef78f77b", C0=320, C1=192, CPOOL=768, OUTC=1280, HIN=17, WIN=17, H=8, W=8, BLOCK=256)
@oracle_impl(hardware="B200", point="faf632a3", C0=384, C1=96, CPOOL=288, OUTC=768, HIN=35, WIN=35, H=17, W=17, BLOCK=256)
def oracle_forward(
    inputs,
    *,
    C0: int,
    C1: int,
    CPOOL: int,
    OUTC: int,
    HIN: int,
    WIN: int,
    H: int,
    W: int,
    BLOCK: int,
):
    (
        pool_in,
        mean0,
        branch0,
        var0,
        weight0,
        bias0,
        mean1,
        branch1,
        var1,
        weight1,
        bias1,
        *_,
    ) = inputs
    total = 128 * OUTC * H * W
    cat = torch.empty_strided((128, OUTC, H, W), (OUTC * H * W, 1, W * OUTC, OUTC), device=pool_in.device, dtype=torch.bfloat16)
    _cat_kernel[(triton.cdiv(total, BLOCK),)](
        pool_in,
        mean0,
        branch0,
        var0,
        weight0,
        bias0,
        mean1,
        branch1,
        var1,
        weight1,
        bias1,
        cat,
        TOTAL=total,
        C0=C0,
        C1=C1,
        CPOOL=CPOOL,
        OUTC=OUTC,
        HIN=HIN,
        WIN=WIN,
        H=H,
        W=W,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=3,
    )
    avg = torch.empty_strided((128, OUTC, H, W), (OUTC * H * W, 1, W * OUTC, OUTC), device=pool_in.device, dtype=torch.bfloat16)
    _avgpool_kernel[(triton.cdiv(total, BLOCK),)](
        cat,
        avg,
        TOTAL=total,
        OUTC=OUTC,
        H=H,
        W=W,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return cat, avg
