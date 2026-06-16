"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete VisFormer average-pool-backward and BN-backward-style return tuple by treating the zero-filled `as_strided_scatter -> as_strided -> expand -> div` chain as a structured broadcast from the pooled bf16 `[128,768]` source, reducing the two f32 channel summaries directly from that source and the channels-last activation, and emitting both the returned contiguous f32 dense epilogue and its bf16 cast without materializing the expanded pool-gradient tensor. Inductor currently lowers the scatter/expand producer, sibling reductions, dependent dense f32 epilogue, and bf16 cast as generic regions around large materialized intermediates; it cannot do this today because scheduler/codegen does not recognize this fixed average-pool-backward scatter as a structured producer that can feed reductions and side outputs while preserving the explicit f32 divide and bf16 source cast boundary. The fix is SCATTER_REDUCE: add a structured average-pool-backward lowering that shares the pooled source across channel reductions and dense consumers, preserves the literal divide-by-49 and reduction scale, and writes the full return scope directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 768
H = 7
W = 7
HW = H * W
DIVISOR = 49.0
REDUCTION_SCALE = 0.00015943877551020407
BLOCK_N = 128


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
def _pool_bn_reduce_hw_kernel(
    pooled_ptr,
    activation_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    act_stride_n: tl.constexpr,
    act_stride_c: tl.constexpr,
    act_stride_h: tl.constexpr,
    act_stride_w: tl.constexpr,
    C_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    DIVISOR_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    n_idx = tl.program_id(1)
    hw = tl.arange(0, BLOCK_HW)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    h = hw // W_
    w = hw - h * W_
    mask = (hw[:, None] < HW_) & (c[None, :] < C_)

    act_offsets = (
        n_idx * act_stride_n
        + c[None, :] * act_stride_c
        + h[:, None] * act_stride_h
        + w[:, None] * act_stride_w
    )
    activation = tl.load(activation_ptr + act_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    pooled = tl.load(
        pooled_ptr + n_idx * C_ + c, mask=c < C_, other=0.0
    ).to(tl.float32)
    divisor = tl.full((BLOCK_HW, BLOCK_C), DIVISOR_, tl.float32)
    pool_grad = _f32_div(pooled[None, :], divisor)

    partial_sum = tl.sum(tl.where(mask, pool_grad, 0.0), axis=0)
    partial_dot = tl.sum(
        tl.where(mask, _f32_mul(pool_grad, activation), 0.0), axis=0
    )
    out_offsets = n_idx * C_ + c
    tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=c < C_)
    tl.store(partial_dot_ptr + out_offsets, partial_dot, mask=c < C_)


@triton.jit
def _pool_bn_finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    stats_ptr,
    C_: tl.constexpr,
    N_: tl.constexpr,
    REDUCTION_SCALE_: tl.constexpr,
    BLOCK_N_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    n = tl.arange(0, BLOCK_N_)[:, None]
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (n < N_) & (c < C_)
    offsets = n * C_ + c

    partial_sum = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    partial_dot = tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_value = tl.sum(partial_sum, axis=0)
    dot_value = tl.sum(partial_dot, axis=0)

    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < C_
    invstd = tl.load(invstd_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    mean_term = _f32_mul(sum_value, REDUCTION_SCALE_)
    dot_scaled = _f32_mul(dot_value, REDUCTION_SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(sum_out_ptr + channels, sum_value, mask=channel_mask)
    tl.store(scale_grad_ptr + channels, _f32_mul(dot_value, invstd), mask=channel_mask)
    tl.store(stats_ptr + channels, mean_term, mask=channel_mask)
    tl.store(stats_ptr + C_ + channels, variance_term, mask=channel_mask)
    tl.store(stats_ptr + 2 * C_ + channels, output_scale, mask=channel_mask)


@triton.jit
def _pool_bn_epilogue_kernel(
    pooled_ptr,
    activation_ptr,
    stats_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    act_stride_n: tl.constexpr,
    act_stride_c: tl.constexpr,
    act_stride_h: tl.constexpr,
    act_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    C_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    DIVISOR_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    n_idx = tl.program_id(1)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    hw = tl.arange(0, BLOCK_HW)[None, :]
    h = hw // W_
    w = hw - h * W_
    mask = (c < C_) & (hw < HW_)

    act_offsets = (
        n_idx * act_stride_n
        + c * act_stride_c
        + h * act_stride_h
        + w * act_stride_w
    )
    out_offsets = (
        n_idx * out_stride_n
        + c * out_stride_c
        + h * out_stride_h
        + w * out_stride_w
    )
    activation = tl.load(activation_ptr + act_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    pooled = tl.load(
        pooled_ptr + n_idx * C_ + c, mask=c < C_, other=0.0
    ).to(tl.float32)
    divisor = tl.full((BLOCK_C, BLOCK_HW), DIVISOR_, tl.float32)
    pool_grad = _f32_div(pooled, divisor)

    mean_term = tl.load(stats_ptr + c, mask=c < C_, other=0.0).to(tl.float32)
    variance_term = tl.load(stats_ptr + C_ + c, mask=c < C_, other=0.0).to(
        tl.float32
    )
    output_scale = tl.load(stats_ptr + 2 * C_ + c, mask=c < C_, other=0.0).to(
        tl.float32
    )

    after_variance = _f32_sub(pool_grad, _f32_mul(activation, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    out_f32 = _f32_mul(after_mean, output_scale)
    tl.store(out_f32_ptr + out_offsets, out_f32, mask=mask)
    tl.store(
        out_bf16_ptr + out_offsets,
        out_f32.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


# timm_visformer_small train, pooled bf16 [128,768] expanded over channels-last [128,768,7,7].
@oracle_impl(hardware="B200", point="09c2ade4", BLOCK_HW=64, BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int):
    pooled, activation, invstd, weight, *_shape_params = inputs
    device = activation.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out_f32 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.float32,
    )
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty_strided((N, C), (C, 1), device=device, dtype=torch.float32)
    partial_dot = torch.empty_strided((N, C), (C, 1), device=device, dtype=torch.float32)
    stats = torch.empty_strided((3, C), (C, 1), device=device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), N)
    _pool_bn_reduce_hw_kernel[grid](
        pooled,
        activation,
        partial_sum,
        partial_dot,
        act_stride_n=activation.stride(0),
        act_stride_c=activation.stride(1),
        act_stride_h=activation.stride(2),
        act_stride_w=activation.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        DIVISOR_=DIVISOR,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _pool_bn_finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        scale_grad,
        stats,
        C_=C,
        N_=N,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_N_=BLOCK_N,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _pool_bn_epilogue_kernel[grid](
        pooled,
        activation,
        stats,
        out_f32,
        out_bf16,
        act_stride_n=activation.stride(0),
        act_stride_c=activation.stride(1),
        act_stride_h=activation.stride(2),
        act_stride_w=activation.stride(3),
        out_stride_n=out_f32.stride(0),
        out_stride_c=out_f32.stride(1),
        out_stride_h=out_f32.stride(2),
        out_stride_w=out_f32.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        DIVISOR_=DIVISOR,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    return sum_out, out_f32, scale_grad, out_bf16
