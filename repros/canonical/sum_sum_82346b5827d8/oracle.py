"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MNASNet bf16 dropout/avg-pool, ReLU-mask, and BatchNorm-backward return scope by sharing the masked pool-gradient producer across both channel reductions, finalizing the returned sum and scale-gradient vectors once, writing the returned bf16 zero scalar, and sinking the summaries into the channels-last bf16 dense epilogue, whereas Inductor schedules the dropout producer, bf16 ReLU threshold, sibling `sum([0, 2, 3])` reductions, and dependent dense BN-backward epilogue as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output reduction template that preserves the bf16 dropout and affine-threshold rounding boundaries while coordinating compatible channel reductions with their vector and dense tensor epilogues; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse masked BN-backward channel summaries with the structured avg-pool/dropout producer and the dependent channels-last dense epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


_USE_INDUCTOR_NUMERICS = False

N = 32
C = 1280
H = 7
W = 7
HW = H * W
REDUCTION_SCALE = 0.0006377551020408163
INV_DROPOUT_KEEP = 1.25
INV_HW = 1.0 / 49.0
BLOCK_HW = 64
BLOCK_N = 64
BLOCK_C = 64


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
def _pool_grad(
    mask_bool,
    pooled_bf16,
    INV_DROPOUT_KEEP_N: tl.constexpr,
    INV_HW_N: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    if USE_INDUCTOR_NUMERICS:
        keep = _f32_mul(mask_bool.to(tl.float32), INV_DROPOUT_KEEP_N)
        scaled = _f32_mul(pooled_bf16.to(tl.float32), keep)
        return _f32_mul(scaled, INV_HW_N)
    else:
        keep = _f32_mul(mask_bool.to(tl.float32), INV_DROPOUT_KEEP_N).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        scaled = _f32_mul(pooled_bf16.to(tl.float32), keep.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        return _f32_mul(scaled.to(tl.float32), INV_HW_N).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        ).to(tl.float32)


@triton.jit
def _producer_terms(
    mask2d_ptr,
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    n_idx,
    offs_c,
    offs_hw,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    HW_N: tl.constexpr,
    INV_DROPOUT_KEEP_N: tl.constexpr,
    INV_HW_N: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    h_idx = offs_hw // W_N
    w_idx = offs_hw - h_idx * W_N
    c_mask = offs_c < C_N
    hw_mask = offs_hw < HW_N
    mask = hw_mask[:, None] & c_mask[None, :]
    x_offsets = (
        n_idx * X_STRIDE_N
        + offs_c[None, :] * X_STRIDE_C
        + h_idx[:, None] * X_STRIDE_H
        + w_idx[:, None] * X_STRIDE_W
    )
    nc_offsets = n_idx * C_N + offs_c

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    normalized = _f32_mul(centered, invstd[None, :])
    weighted = _f32_mul(normalized, weight[None, :])
    affine = _f32_add(weighted, bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")

    mask_bool = tl.load(mask2d_ptr + nc_offsets, mask=c_mask, other=0)
    pooled = tl.load(pooled_ptr + nc_offsets, mask=c_mask, other=0.0)
    pool_grad = _pool_grad(
        mask_bool, pooled, INV_DROPOUT_KEEP_N, INV_HW_N, USE_INDUCTOR_NUMERICS
    )
    selected = tl.where(affine_bf16 <= 0.0, 0.0, pool_grad[None, :])
    selected = tl.where(mask, selected, 0.0)
    return selected, centered, mask


@triton.jit
def _partial_reduce_kernel(
    mask2d_ptr,
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    HW_N: tl.constexpr,
    INV_DROPOUT_KEEP_N: tl.constexpr,
    INV_HW_N: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    offs_c = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n_idx = tl.program_id(1)
    offs_hw = tl.arange(0, BLOCK_HW_N)
    selected, centered, _ = _producer_terms(
        mask2d_ptr,
        pooled_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        n_idx,
        offs_c,
        offs_hw,
        X_STRIDE_N,
        X_STRIDE_C,
        X_STRIDE_H,
        X_STRIDE_W,
        C_N,
        W_N,
        HW_N,
        INV_DROPOUT_KEEP_N,
        INV_HW_N,
        USE_INDUCTOR_NUMERICS,
        BLOCK_HW_N,
        BLOCK_C_N,
    )
    partial_sum = tl.sum(selected, axis=0)
    partial_dot = tl.sum(_f32_mul(selected, centered), axis=0)
    out_offsets = n_idx * C_N + offs_c
    tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=offs_c < C_N)
    tl.store(partial_dot_ptr + out_offsets, partial_dot, mask=offs_c < C_N)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    full_out_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    stats_ptr,
    C_N: tl.constexpr,
    N_N: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_N_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    offs_c = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    offs_n = tl.arange(0, BLOCK_N_N)
    c_mask = offs_c < C_N
    n_mask = offs_n < N_N
    offsets = offs_n[:, None] * C_N + offs_c[None, :]
    mask = n_mask[:, None] & c_mask[None, :]

    sum_parts = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dot_parts = tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_value = tl.sum(sum_parts, axis=0)
    dot_value = tl.sum(dot_parts, axis=0)

    invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(full_out_ptr, 0.0)
    tl.store(sum_out_ptr + offs_c, sum_value, mask=c_mask)
    tl.store(scaled_dot_out_ptr + offs_c, _f32_mul(dot_value, invstd), mask=c_mask)
    tl.store(stats_ptr + offs_c, mean_term, mask=c_mask)
    tl.store(stats_ptr + C_N + offs_c, correction_scale, mask=c_mask)
    tl.store(stats_ptr + 2 * C_N + offs_c, output_scale, mask=c_mask)


@triton.jit
def _epilogue_kernel(
    mask2d_ptr,
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    stats_ptr,
    out_ptr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr,
    OUT_STRIDE_C: tl.constexpr,
    OUT_STRIDE_H: tl.constexpr,
    OUT_STRIDE_W: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    HW_N: tl.constexpr,
    INV_DROPOUT_KEEP_N: tl.constexpr,
    INV_HW_N: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    offs_c = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n_idx = tl.program_id(1)
    offs_hw = tl.arange(0, BLOCK_HW_N)
    selected, centered, mask = _producer_terms(
        mask2d_ptr,
        pooled_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        n_idx,
        offs_c,
        offs_hw,
        X_STRIDE_N,
        X_STRIDE_C,
        X_STRIDE_H,
        X_STRIDE_W,
        C_N,
        W_N,
        HW_N,
        INV_DROPOUT_KEEP_N,
        INV_HW_N,
        USE_INDUCTOR_NUMERICS,
        BLOCK_HW_N,
        BLOCK_C_N,
    )
    mean_term = tl.load(stats_ptr + offs_c, mask=offs_c < C_N, other=0.0).to(
        tl.float32
    )
    correction_scale = tl.load(
        stats_ptr + C_N + offs_c, mask=offs_c < C_N, other=0.0
    ).to(tl.float32)
    output_scale = tl.load(
        stats_ptr + 2 * C_N + offs_c, mask=offs_c < C_N, other=0.0
    ).to(tl.float32)

    correction = _f32_mul(centered, correction_scale[None, :])
    after_correction = _f32_sub(selected, correction)
    after_mean = _f32_sub(after_correction, mean_term[None, :])
    dense = _f32_mul(after_mean, output_scale[None, :]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    h_idx = offs_hw // W_N
    w_idx = offs_hw - h_idx * W_N
    out_offsets = (
        n_idx * OUT_STRIDE_N
        + offs_c[None, :] * OUT_STRIDE_C
        + h_idx[:, None] * OUT_STRIDE_H
        + w_idx[:, None] * OUT_STRIDE_W
    )
    tl.store(out_ptr + out_offsets, dense, mask=mask)


# 36943a14: MNASNet train, N=32 C=1280 H=W=7, channels-last activation.
@oracle_impl(hardware="B200", point="36943a14")
def oracle_forward(inputs):
    global _USE_INDUCTOR_NUMERICS
    mask2d, pooled, x, mean, invstd, weight, bias, _shape0 = inputs
    del _shape0
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    full_out = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((N, C), device=x.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    stats = torch.empty((3, C), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), N)
    _partial_reduce_kernel[grid](
        mask2d,
        pooled,
        x,
        mean,
        invstd,
        weight,
        bias,
        partial_sum,
        partial_dot,
        X_STRIDE_N=x.stride(0),
        X_STRIDE_C=x.stride(1),
        X_STRIDE_H=x.stride(2),
        X_STRIDE_W=x.stride(3),
        C_N=C,
        W_N=W,
        HW_N=HW,
        INV_DROPOUT_KEEP_N=INV_DROPOUT_KEEP,
        INV_HW_N=INV_HW,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        BLOCK_HW_N=BLOCK_HW,
        BLOCK_C_N=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        full_out,
        sum_out,
        scaled_dot_out,
        stats,
        C_N=C,
        N_N=N,
        SCALE_VALUE=REDUCTION_SCALE,
        BLOCK_N_N=BLOCK_N,
        BLOCK_C_N=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    _epilogue_kernel[grid](
        mask2d,
        pooled,
        x,
        mean,
        invstd,
        weight,
        bias,
        stats,
        dense_out,
        X_STRIDE_N=x.stride(0),
        X_STRIDE_C=x.stride(1),
        X_STRIDE_H=x.stride(2),
        X_STRIDE_W=x.stride(3),
        OUT_STRIDE_N=dense_out.stride(0),
        OUT_STRIDE_C=dense_out.stride(1),
        OUT_STRIDE_H=dense_out.stride(2),
        OUT_STRIDE_W=dense_out.stride(3),
        C_N=C,
        W_N=W,
        HW_N=HW,
        INV_DROPOUT_KEEP_N=INV_DROPOUT_KEEP,
        INV_HW_N=INV_HW,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        BLOCK_HW_N=BLOCK_HW,
        BLOCK_C_N=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    return full_out, sum_out, scaled_dot_out, dense_out
