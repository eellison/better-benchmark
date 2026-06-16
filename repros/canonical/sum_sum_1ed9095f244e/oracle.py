"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV2 bf16 adaptive-average-pool-backward plus hard-ReLU6-mask and BatchNorm-backward return tuple, including the returned bf16 zero scalar, both f32 per-channel reductions, and the channels-last bf16 dense epilogue. Inductor currently lowers the zero-fill/as_strided_scatter/expand pool-gradient producer, bf16 affine threshold, sibling reductions, and reduction-dependent BN-backward epilogue as separate generic regions over materialized intermediates. It cannot do this today because scheduler/codegen does not model the fixed average-pool backward producer as a structured source that can be shared by compatible channel reductions and their dependent full-tensor consumer while preserving the bf16 division, affine thresholding, and output cast boundaries. The fix is SCHEDULER_FUSION: add a guarded pool-backward BN-backward template that keeps the masked producer live, finalizes the two channel summaries once, and sinks those summaries into the channels-last epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 1280
H = 7
W = 7
HW = H * W
REDUCTION_SCALE = 0.00015943877551020407
POOL_DIVISOR = 49.0


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
def _producer_terms(
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    n_idx,
    offs_c,
    offs_hw,
    X_SN: tl.constexpr,
    X_SC: tl.constexpr,
    X_SH: tl.constexpr,
    X_SW: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    HW_N: tl.constexpr,
    POOL_DIVISOR_VALUE: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    h_idx = offs_hw // W_N
    w_idx = offs_hw - h_idx * W_N
    c_mask = offs_c < C_N
    hw_mask = offs_hw < HW_N
    mask = hw_mask[:, None] & c_mask[None, :]

    x_offsets = (
        n_idx * X_SN
        + offs_c[None, :] * X_SC
        + h_idx[:, None] * X_SH
        + w_idx[:, None] * X_SW
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

    pooled = tl.load(pooled_ptr + nc_offsets, mask=c_mask, other=0.0).to(tl.float32)
    pool_grad = _f32_div(pooled, POOL_DIVISOR_VALUE).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    active = (affine_bf16 > 0.0) & (affine_bf16 < 6.0) & mask
    selected = tl.where(active, pool_grad[None, :], 0.0)
    return selected, centered, mask, x_offsets


@triton.jit
def _partial_reduce_kernel(
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    where_tmp_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    X_SN: tl.constexpr,
    X_SC: tl.constexpr,
    X_SH: tl.constexpr,
    X_SW: tl.constexpr,
    TMP_SN: tl.constexpr,
    TMP_SC: tl.constexpr,
    TMP_SH: tl.constexpr,
    TMP_SW: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    HW_N: tl.constexpr,
    POOL_DIVISOR_VALUE: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    offs_c = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n_idx = tl.program_id(1)
    offs_hw = tl.arange(0, BLOCK_HW_N)
    selected, centered, _mask, _ = _producer_terms(
        pooled_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        n_idx,
        offs_c,
        offs_hw,
        X_SN,
        X_SC,
        X_SH,
        X_SW,
        C_N,
        W_N,
        HW_N,
        POOL_DIVISOR_VALUE,
        BLOCK_HW_N,
        BLOCK_C_N,
    )
    partial_sum = tl.sum(selected, axis=0)
    partial_dot = tl.sum(_f32_mul(selected, centered), axis=0)
    offsets = n_idx * C_N + offs_c
    tl.store(partial_sum_ptr + offsets, partial_sum, mask=offs_c < C_N)
    tl.store(partial_dot_ptr + offsets, partial_dot, mask=offs_c < C_N)

    h_idx = offs_hw // W_N
    w_idx = offs_hw - h_idx * W_N
    tmp_offsets = (
        n_idx * TMP_SN
        + offs_c[None, :] * TMP_SC
        + h_idx[:, None] * TMP_SH
        + w_idx[:, None] * TMP_SW
    )
    tmp_value = selected.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tmp_mask = (offs_hw[:, None] < HW_N) & (offs_c[None, :] < C_N)
    tl.store(where_tmp_ptr + tmp_offsets, tmp_value, mask=tmp_mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    zero_out_ptr,
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

    tl.store(zero_out_ptr, 0.0, mask=tl.program_id(0) == 0)
    tl.store(sum_out_ptr + offs_c, sum_value, mask=c_mask)
    tl.store(scaled_dot_out_ptr + offs_c, _f32_mul(dot_value, invstd), mask=c_mask)
    tl.store(stats_ptr + offs_c, mean_term, mask=c_mask)
    tl.store(stats_ptr + C_N + offs_c, correction_scale, mask=c_mask)
    tl.store(stats_ptr + 2 * C_N + offs_c, output_scale, mask=c_mask)


@triton.jit
def _epilogue_kernel(
    where_tmp_ptr,
    x_ptr,
    mean_ptr,
    stats_ptr,
    out_ptr,
    X_SN: tl.constexpr,
    X_SC: tl.constexpr,
    X_SH: tl.constexpr,
    X_SW: tl.constexpr,
    TMP_SN: tl.constexpr,
    TMP_SC: tl.constexpr,
    TMP_SH: tl.constexpr,
    TMP_SW: tl.constexpr,
    OUT_SN: tl.constexpr,
    OUT_SC: tl.constexpr,
    OUT_SH: tl.constexpr,
    OUT_SW: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    HW_N: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    offs_c = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n_idx = tl.program_id(1)
    offs_hw = tl.arange(0, BLOCK_HW_N)
    h_idx = offs_hw // W_N
    w_idx = offs_hw - h_idx * W_N
    mask = (offs_hw[:, None] < HW_N) & (offs_c[None, :] < C_N)
    x_offsets = (
        n_idx * X_SN
        + offs_c[None, :] * X_SC
        + h_idx[:, None] * X_SH
        + w_idx[:, None] * X_SW
    )
    tmp_offsets = (
        n_idx * TMP_SN
        + offs_c[None, :] * TMP_SC
        + h_idx[:, None] * TMP_SH
        + w_idx[:, None] * TMP_SW
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + offs_c, mask=offs_c < C_N, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean[None, :])
    selected = tl.load(where_tmp_ptr + tmp_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + offs_c, mask=offs_c < C_N, other=0.0).to(tl.float32)
    correction_scale = tl.load(
        stats_ptr + C_N + offs_c, mask=offs_c < C_N, other=0.0
    ).to(tl.float32)
    output_scale = tl.load(
        stats_ptr + 2 * C_N + offs_c, mask=offs_c < C_N, other=0.0
    ).to(tl.float32)

    centered_mul = _f32_mul(centered, correction_scale[None, :])
    sub_centered = _f32_sub(selected, centered_mul)
    sub_mean = _f32_sub(sub_centered, mean_term[None, :])
    dense = _f32_mul(sub_mean, output_scale[None, :]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    out_offsets = (
        n_idx * OUT_SN
        + offs_c[None, :] * OUT_SC
        + h_idx[:, None] * OUT_SH
        + w_idx[:, None] * OUT_SW
    )
    tl.store(out_ptr + out_offsets, dense, mask=mask)


@oracle_impl(
    hardware="B200",
    point="a44186d7",
    BLOCK_HW=64,
    BLOCK_C=32,
    BLOCK_N=128,
    reduce_warps=4,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    BLOCK_N: int,
    reduce_warps: int,
    final_warps: int,
):
    (
        pooled,
        x,
        mean,
        invstd,
        weight,
        bias,
        _shape0,
        _shape1,
        _shape2,
        _shape3,
        _shape4,
        _shape5,
        _shape6,
    ) = inputs

    zero_out = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    where_tmp = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty_strided((N, C), (C, 1), device=x.device, dtype=torch.float32)
    partial_dot = torch.empty_strided((N, C), (C, 1), device=x.device, dtype=torch.float32)
    stats = torch.empty_strided((3, C), (C, 1), device=x.device, dtype=torch.float32)

    x_stride = tuple(int(s) for s in x.stride())
    tmp_stride = tuple(int(s) for s in where_tmp.stride())
    out_stride = tuple(int(s) for s in dense_out.stride())
    grid = (triton.cdiv(C, BLOCK_C), N)
    _partial_reduce_kernel[grid](
        pooled,
        x,
        mean,
        invstd,
        weight,
        bias,
        where_tmp,
        partial_sum,
        partial_dot,
        X_SN=x_stride[0],
        X_SC=x_stride[1],
        X_SH=x_stride[2],
        X_SW=x_stride[3],
        TMP_SN=tmp_stride[0],
        TMP_SC=tmp_stride[1],
        TMP_SH=tmp_stride[2],
        TMP_SW=tmp_stride[3],
        C_N=C,
        W_N=W,
        HW_N=HW,
        POOL_DIVISOR_VALUE=POOL_DIVISOR,
        BLOCK_HW_N=BLOCK_HW,
        BLOCK_C_N=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        zero_out,
        sum_out,
        scaled_dot_out,
        stats,
        C_N=C,
        N_N=N,
        SCALE_VALUE=REDUCTION_SCALE,
        BLOCK_N_N=BLOCK_N,
        BLOCK_C_N=BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )
    _epilogue_kernel[grid](
        where_tmp,
        x,
        mean,
        stats,
        dense_out,
        X_SN=x_stride[0],
        X_SC=x_stride[1],
        X_SH=x_stride[2],
        X_SW=x_stride[3],
        TMP_SN=tmp_stride[0],
        TMP_SC=tmp_stride[1],
        TMP_SH=tmp_stride[2],
        TMP_SW=tmp_stride[3],
        OUT_SN=out_stride[0],
        OUT_SC=out_stride[1],
        OUT_SH=out_stride[2],
        OUT_SW=out_stride[3],
        C_N=C,
        W_N=W,
        HW_N=HW,
        BLOCK_HW_N=BLOCK_HW,
        BLOCK_C_N=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    return zero_out, sum_out, scaled_dot_out, dense_out
