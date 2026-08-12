"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 BatchNorm-backward return scope by preserving the bf16 ReLU-gate cast and sliced bf16 producer, split-K reducing the shared gated tensor into both returned f32 channel summaries, and sinking the finalized summaries into a channels-last bf16 dense epilogue, whereas Inductor schedules the gate, sibling reductions, and reduction-dependent dense epilogue as separate generic pointwise/reduction regions; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output BN-backward template that coordinates the bf16 gated producer, two per-channel summaries, vector side output, and dense epilogue while preserving bf16 cast boundaries and output strides; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward split-K template that fuses compatible producer, channel reductions, vector side output, and dense epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_IN = 16
C = 8
H = 112
W = 112
HW = H * W
K_TOTAL = N * HW
TOTAL = N * C * HW
SCALE = 1.5570192920918366e-07


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
def _reduce_store_kernel(
    slice_base_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
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
    source_offsets = (
        n[:, None] * (C_IN_ * HW_)
        + h[:, None] * (W_ * C_IN_)
        + w[:, None] * C_IN_
        + (C_ + c[None, :])
    )
    compact_offsets = (
        n[:, None] * (C_ * HW_)
        + h[:, None] * (W_ * C_)
        + w[:, None] * C_
        + c[None, :]
    )

    source = tl.load(slice_base_ptr + source_offsets, mask=active, other=0.0)
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
    selected_bf16 = tl.where(affine_bf16 <= 0.0, fill, source)

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
    slice_base_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    sum_ptr,
    dot_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    TOTAL_: tl.constexpr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL_
    c = linear % C_
    hw = (linear // C_) % HW_
    n = linear // (C_ * HW_)
    h = hw // W_
    w = hw - h * W_

    source_offsets = n * (C_IN_ * HW_) + h * (W_ * C_IN_) + w * C_IN_ + (C_ + c)
    source = tl.load(slice_base_ptr + source_offsets, mask=active, other=0.0)
    x = tl.load(bn_input_ptr + linear, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected = tl.where(affine_bf16 <= 0.0, fill, source).to(tl.float32)

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
    point="a67efa72",
    BLOCK_K=1024,
    BLOCK_C=8,
    ZERO_BLOCK=8,
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
    slice_base, bn_input, mean, invstd, weight, bias, fill = inputs
    num_k_tiles = triton.cdiv(K_TOTAL, BLOCK_K)
    device = bn_input.device

    dense_out = torch.empty_strided(
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
    _reduce_store_kernel[(triton.cdiv(C, BLOCK_C), num_k_tiles)](
        slice_base,
        bn_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
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
        slice_base,
        bn_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        sum_out,
        dot_out,
        scale_grad,
        dense_out,
        TOTAL_=TOTAL,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        W_=W,
        SCALE_=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out
