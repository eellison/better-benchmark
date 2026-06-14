"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ShuffleNet bf16 channel-shuffle plus BatchNorm-backward return scope by preserving the bf16 add/view/permute/clone shuffle output, split-K reducing the shared ReLU-gated high-half producer into both returned f32 channel summaries, and sinking the finalized summaries into the channels-last bf16 dense-gradient epilogue. Inductor currently schedules the channel shuffle, bf16 ReLU gate, sibling reductions, and reduction-dependent BN-backward epilogue as separate generic pointwise and reduction regions around materialized intermediates; it cannot do this today because scheduler/codegen has no cooperative split-K multi-output template that coordinates layout-changing producers with compatible channel reductions and their dependent dense/vector epilogues while preserving bf16 cast boundaries and output strides. The fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN-backward channel reductions across the reduced N/H/W domain, combine partial summaries once, and fuse the channel-shuffle producer plus downstream tensor/vector epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C_IN = 116
C = 58
H = 28
W = 28
HW = H * W
K_TOTAL = N * HW
TOTAL = N * C * HW
SCALE = 9.964923469387754e-06


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _shuffle_low_kernel(
    lhs_ptr,
    rhs_ptr,
    shuffle_out_ptr,
    sum_ptr,
    dot_ptr,
    TOTAL_: tl.constexpr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL_
    hw = linear % HW_
    c = (linear // HW_) % C_
    n = linear // (C_ * HW_)
    h = hw // W_
    w = hw - h * W_
    input_c = c * 2

    input_offsets = n * (C_IN_ * HW_) + h * (W_ * C_IN_) + w * C_IN_ + input_c
    output_offsets = n * (C_IN_ * HW_) + c * HW_ + hw
    lhs = tl.load(lhs_ptr + input_offsets, mask=active, other=0.0)
    rhs = tl.load(rhs_ptr + input_offsets, mask=active, other=0.0)
    tl.store(shuffle_out_ptr + output_offsets, _bf16_add(lhs, rhs), mask=active)
    tl.store(sum_ptr + linear, 0.0, mask=linear < C_)
    tl.store(dot_ptr + linear, 0.0, mask=linear < C_)


@triton.jit
def _zero_kernel(
    sum_ptr,
    dot_ptr,
    C_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < C_
    tl.store(sum_ptr + offsets, 0.0, mask=mask)
    tl.store(dot_ptr + offsets, 0.0, mask=mask)


@triton.jit
def _partial_reduce_high_kernel(
    lhs_ptr,
    rhs_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    shuffle_out_ptr,
    selected_ptr,
    sum_ptr,
    dot_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = (k[:, None] < K_TOTAL_) & (c[None, :] < C_)

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    input_c = c * 2 + 1
    input_offsets = (
        n[:, None] * (C_IN_ * HW_)
        + h[:, None] * (W_ * C_IN_)
        + w[:, None] * C_IN_
        + input_c[None, :]
    )
    compact_offsets = (
        n[:, None] * (C_ * HW_)
        + h[:, None] * (W_ * C_)
        + w[:, None] * C_
        + c[None, :]
    )
    lhs = tl.load(lhs_ptr + input_offsets, mask=active, other=0.0)
    rhs = tl.load(rhs_ptr + input_offsets, mask=active, other=0.0)
    source_bf16 = _bf16_add(lhs, rhs)
    low_lhs = tl.load(lhs_ptr + input_offsets - 1, mask=active, other=0.0)
    low_rhs = tl.load(rhs_ptr + input_offsets - 1, mask=active, other=0.0)
    low_out_offsets = n[:, None] * (C_IN_ * HW_) + c[None, :] * HW_ + hw[:, None]
    high_out_offsets = (
        n[:, None] * (C_IN_ * HW_) + (C_ + c[None, :]) * HW_ + hw[:, None]
    )
    tl.store(shuffle_out_ptr + low_out_offsets, _bf16_add(low_lhs, low_rhs), mask=active)
    tl.store(shuffle_out_ptr + high_out_offsets, source_bf16, mask=active)

    x = tl.load(bn_input_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c < C_, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c < C_, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < C_, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c < C_, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr)

    centered = _f32_sub(x, mean[None, :])
    normalized = _f32_mul(centered, invstd[None, :])
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected_bf16 = tl.where(affine_bf16 <= 0.0, fill, source_bf16)
    tl.store(selected_ptr + compact_offsets, selected_bf16, mask=active)
    selected = selected_bf16.to(tl.float32)

    selected = tl.where(active, selected, 0.0)
    dot = tl.where(active, _f32_mul(selected, centered), 0.0)
    tl.atomic_add(
        sum_ptr + c,
        tl.sum(selected, axis=0),
        mask=c < C_,
        sem="relaxed",
    )
    tl.atomic_add(
        dot_ptr + c,
        tl.sum(dot, axis=0),
        mask=c < C_,
        sem="relaxed",
    )


@triton.jit
def _epilogue_kernel(
    selected_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    TOTAL_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL_
    c = linear % C_
    hw = (linear // C_) % HW_
    n = linear // (C_ * HW_)

    selected = tl.load(selected_ptr + linear, mask=active, other=0.0).to(tl.float32)
    x = tl.load(bn_input_ptr + linear, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)

    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = _f32_mul(sum_value, SCALE_)
    dot_scaled = _f32_mul(dot_value, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    after_variance = _f32_sub(selected, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    out = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(dense_out_ptr + linear, out, mask=active)
    tl.store(
        scale_grad_ptr + c,
        _f32_mul(dot_value, invstd),
        mask=active & (n == 0) & (hw == 0),
    )


@oracle_impl(
    hardware="B200",
    point="aa052f71",
    BLOCK_K=256,
    BLOCK_C=8,
    ZERO_BLOCK=64,
    EPILOGUE_BLOCK=512,
    reduce_warps=4,
    epilogue_warps=2,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    ZERO_BLOCK: int,
    EPILOGUE_BLOCK: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    lhs, rhs, bn_input, mean, invstd, weight, bias, fill, _shape0, _shape1 = inputs
    num_k_tiles = triton.cdiv(K_TOTAL, BLOCK_K)
    device = bn_input.device

    shuffle_out = torch.empty_strided(
        (N, C_IN, H, W),
        (C_IN * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    selected_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)

    _zero_kernel[(1,)](
        sum_out,
        dot_out,
        C_=C,
        BLOCK=ZERO_BLOCK,
        num_warps=1,
    )
    _partial_reduce_high_kernel[(triton.cdiv(C, BLOCK_C), num_k_tiles)](
        lhs,
        rhs,
        bn_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        shuffle_out,
        selected_out,
        sum_out,
        dot_out,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        W_=W,
        K_TOTAL_=K_TOTAL,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=4,
    )
    _epilogue_kernel[(triton.cdiv(TOTAL, EPILOGUE_BLOCK),)](
        selected_out,
        bn_input,
        mean,
        invstd,
        weight,
        sum_out,
        dot_out,
        scale_grad,
        dense_out,
        TOTAL_=TOTAL,
        C_=C,
        HW_=HW,
        SCALE_=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return shuffle_out, sum_out, scale_grad, dense_out
