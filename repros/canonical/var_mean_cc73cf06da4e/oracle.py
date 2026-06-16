"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT bf16 training-BatchNorm SiLU channel-concat scope, including population var_mean over the channels-last input, eps=1e-5 rsqrt, in-place running mean/variance copy_ returns with the captured variance correction, the explicit bf16 affine rounding before natural-exp SiLU, and the returned channels-last bf16 cat tensor, whereas Inductor lowers the stats reduction, mutable running-stat epilogue, bf16 rounding boundary, SiLU pointwise, and cat store as generic separated work; Inductor cannot do this today because its norm-template scheduler does not keep the multi-output BN-training statistics, side-effecting running-stat updates, activation consumer, and concatenation layout resident in one full-scope channel-tiled plan; the fix is SCHEDULER_FUSION: teach the BN-training scheduler to fuse channel-reduction side outputs, mutable copy_ updates, explicit bf16 activation rounding, natural-exp SiLU, and direct channels-last cat emission."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _partial_sum_sumsq_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    hw = H * W
    n_offsets = e_offsets // hw
    hw_offsets = e_offsets - n_offsets * hw
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets - h_offsets * W
    mask = (c_offsets[:, None] < C) & (e_offsets[None, :] < E)
    x_offsets = (
        n_offsets[None, :] * x_s0
        + c_offsets[:, None] * x_s1
        + h_offsets[None, :] * x_s2
        + w_offsets[None, :] * x_s3
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(x, axis=1)
    sumsq = tl.sum(x * x, axis=1)
    partial_offsets = tl.program_id(1) * C + c_offsets
    tl.store(partial_sum_ptr + partial_offsets, sums, mask=c_offsets < C)
    tl.store(partial_sumsq_ptr + partial_offsets, sumsq, mask=c_offsets < C)


@triton.jit
def _finalize_stats_kernel(
    partial_sum_ptr,
    partial_sumsq_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    mask = (c_offsets[:, None] < C) & (split_offsets[None, :] < NUM_SPLITS)
    partial_offsets = split_offsets[None, :] * C + c_offsets[:, None]
    sums = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0)
    sumsq = tl.load(partial_sumsq_ptr + partial_offsets, mask=mask, other=0.0)
    total = tl.sum(sums, axis=1).to(tl.float32)
    total_sumsq = tl.sum(sumsq, axis=1).to(tl.float32)
    mean = total / E
    var = total_sumsq / E - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + 1.0e-5)

    c_mask = c_offsets < C
    old_mean = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(mean_ptr + c_offsets, mean, mask=c_mask)
    tl.store(invstd_ptr + c_offsets, invstd, mask=c_mask)
    tl.store(running_mean_ptr + c_offsets, old_mean * 0.9 + mean * 0.1, mask=c_mask)
    tl.store(
        running_var_ptr + c_offsets,
        old_var * 0.9 + var * 1.0001220852154804 * 0.1,
        mask=c_mask,
    )


@triton.jit
def _silu_cat_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    cat_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    residual_s0: tl.constexpr,
    residual_s1: tl.constexpr,
    residual_s2: tl.constexpr,
    residual_s3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c_offsets = offsets % C
    tmp = offsets // C
    w_offsets = tmp % W
    tmp = tmp // W
    h_offsets = tmp % H
    n_offsets = tmp // H

    x_offsets = (
        n_offsets * x_s0
        + c_offsets * x_s1
        + h_offsets * x_s2
        + w_offsets * x_s3
    )
    residual_offsets = (
        n_offsets * residual_s0
        + c_offsets * residual_s1
        + h_offsets * residual_s2
        + w_offsets * residual_s3
    )
    out_c = 2 * C
    cat_base = n_offsets * (out_c * H * W) + h_offsets * (W * out_c) + w_offsets * out_c + c_offsets

    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0)
    tl.store(cat_ptr + cat_base, residual, mask=mask)

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    affine = ((x - mean) * invstd) * weight + bias
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    silu = rounded / (libdevice.exp(-rounded) + 1.0)
    tl.store(cat_ptr + cat_base + C, silu.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="0be627fe", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=256, stats_warps=4, final_warps=1, out_warps=4)
@oracle_impl(hardware="B200", point="813b1e31", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=256, stats_warps=4, final_warps=1, out_warps=4)
@oracle_impl(hardware="B200", point="fe6f268d", BLOCK_E=1024, C_BLOCK=8, FINAL_C_BLOCK=8, OUT_BLOCK=256, stats_warps=4, final_warps=1, out_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    FINAL_C_BLOCK: int,
    OUT_BLOCK: int,
    stats_warps: int,
    final_warps: int,
    out_warps: int,
):
    x, running_mean, running_var, weight, bias, residual = inputs
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    e = n * h * w
    total = n * c * h * w
    out_c = 2 * c

    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    cat = torch.empty_strided(
        (n, out_c, h, w),
        (out_c * h * w, 1, out_c * w, out_c),
        device=x.device,
        dtype=torch.bfloat16,
    )
    num_splits = triton.cdiv(e, BLOCK_E)
    block_splits = triton.next_power_of_2(num_splits)
    partial_sum = torch.empty((num_splits, c), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty((num_splits, c), device=x.device, dtype=torch.float32)

    _partial_sum_sumsq_kernel[(triton.cdiv(c, C_BLOCK), num_splits)](
        x,
        partial_sum,
        partial_sumsq,
        C=c,
        H=h,
        W=w,
        E=e,
        x_s0=x.stride(0),
        x_s1=x.stride(1),
        x_s2=x.stride(2),
        x_s3=x.stride(3),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=stats_warps,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(c, FINAL_C_BLOCK),)](
        partial_sum,
        partial_sumsq,
        running_mean,
        running_var,
        mean,
        invstd,
        C=c,
        E=e,
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=block_splits,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=final_warps,
        num_stages=3,
    )
    _silu_cat_kernel[(triton.cdiv(total, OUT_BLOCK),)](
        x,
        residual,
        weight,
        bias,
        mean,
        invstd,
        cat,
        C=c,
        H=h,
        W=w,
        TOTAL=total,
        x_s0=x.stride(0),
        x_s1=x.stride(1),
        x_s2=x.stride(2),
        x_s3=x.stride(3),
        residual_s0=residual.stride(0),
        residual_s1=residual.stride(1),
        residual_s2=residual.stride(2),
        residual_s3=residual.stride(3),
        BLOCK=OUT_BLOCK,
        num_warps=out_warps,
        num_stages=3,
    )
    return mean, invstd, cat, running_mean, running_var
