"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Visformer training-BatchNorm tail with the same full scope as the repro, including fp32+bf16 residual promotion, population var_mean over N/H/W, eps=1e-5 rsqrt, saved invstd output, in-place running mean/variance copy_ aliases, channels-last centered activation output, affine normalization, spatial mean, and final bf16 rounding of the pooled [N,C] output; Inductor already lowers this channels-last canonical norm template to the same practical three-kernel Welford statistics, running-stat epilogue, centered activation, and spatial-mean plan; the fix is BANDWIDTH_BOUND: this oracle records the floor for the required residual reads, Welford traffic, centered stores, pooled stores, and launch envelope unless broader normalization codegen or bandwidth work moves both implementations."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


N = 128
C = 768
H = 7
W = 7
HW = H * W

INPUT_SHAPE = (N, C, H, W)
INPUT_STRIDE = (37632, 1, 5376, 768)
VECTOR_SHAPE = (C,)
VECTOR_STRIDE = (1,)
POOLED_SHAPE = (N, C)
POOLED_STRIDE = (C, 1)
PARTIAL_SHAPE = (HW, C)
PARTIAL_STRIDE = (C, 1)


@triton.jit
def _partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    BLOCK_X: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    x_index = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
    r_index = tl.arange(0, BLOCK_R)[None, :]
    x_mask = x_index < 37632
    r_mask = r_index < 128
    channel = x_index % 768
    chunk = x_index // 768

    offsets = channel + 768 * r_index + 98304 * chunk
    x0 = tl.load(x0_ptr + offsets, mask=x_mask & r_mask, other=0.0)
    x1 = tl.load(x1_ptr + offsets, mask=x_mask & r_mask, other=0.0).to(tl.float32)
    values = x0 + x1.to(tl.float32)
    means = tl.where(x_mask & r_mask, values, 0.0)
    m2 = tl.zeros([BLOCK_X, BLOCK_R], tl.float32)
    weights = tl.where(x_mask & r_mask, 1.0, 0.0)
    block_mean, block_m2, _block_weight = triton_helpers.welford(means, m2, weights, 1)

    out_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    out_mask = out_offsets < 37632
    tl.store(partial_mean_ptr + out_offsets, block_mean, mask=out_mask)
    tl.store(partial_m2_ptr + out_offsets, block_m2, mask=out_mask)


@triton.jit
def _finalize_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_out_ptr,
    invstd_out_ptr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    r_index = tl.arange(0, BLOCK_R)[None, :]
    c_mask = channels < 768
    r_mask = r_index < 49
    offsets = channels + 768 * r_index

    partial_mean = tl.load(partial_mean_ptr + offsets, mask=c_mask & r_mask, other=0.0)
    partial_m2 = tl.load(partial_m2_ptr + offsets, mask=c_mask & r_mask, other=0.0)
    mean_inputs = tl.where(c_mask & r_mask, partial_mean, 0.0)
    m2_inputs = tl.where(c_mask & r_mask, partial_m2, 0.0)
    weight_inputs = tl.where(c_mask & r_mask, 128.0, 0.0)
    mean, m2, _weight = triton_helpers.welford(mean_inputs, m2_inputs, weight_inputs, 1)
    var = m2 / 6272.0
    invstd = libdevice.rsqrt(var + 1.0e-5)

    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c_offsets < 768
    old_running_mean = tl.load(running_mean_ptr + c_offsets, mask=mask, other=0.0)
    old_running_var = tl.load(running_var_ptr + c_offsets, mask=mask, other=0.0)
    new_running_mean = mean * 0.1 + old_running_mean * 0.9
    new_running_var = var * 1.0001594642002871 * 0.1 + old_running_var * 0.9

    tl.store(mean_out_ptr + c_offsets, mean, mask=mask)
    tl.store(invstd_out_ptr + c_offsets, invstd, mask=mask)
    tl.store(running_mean_ptr + c_offsets, new_running_mean, mask=mask)
    tl.store(running_var_ptr + c_offsets, new_running_var, mask=mask)


@triton.jit
def _center_pool_kernel(
    x0_ptr,
    x1_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    pooled_out_ptr,
    centered_out_ptr,
    BLOCK_X: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    x_index = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
    r_index = tl.arange(0, BLOCK_R)[None, :]
    r_mask = r_index < 49
    channel = x_index % 768
    batch = x_index // 768

    offsets = channel + 768 * r_index + 37632 * batch
    x0 = tl.load(x0_ptr + offsets, mask=r_mask, other=0.0)
    x1 = tl.load(x1_ptr + offsets, mask=r_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel)
    invstd = tl.load(invstd_ptr + channel)
    weight = tl.load(weight_ptr + channel)
    bias = tl.load(bias_ptr + channel)

    added = x0 + x1.to(tl.float32)
    centered = added - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_sum = tl.sum(tl.where(r_mask, affine, 0.0), 1)[:, None]
    pooled = affine_sum / 49.0

    tl.store(pooled_out_ptr + x_index, pooled, mask=x_index < 98304)
    tl.store(centered_out_ptr + offsets, centered, mask=r_mask)


# d9d8b8eb: (T([128,768,7,7], f32, stride=(37632,1,5376,768)), T([128,768,7,7], bf16, stride=(37632,1,5376,768)), ...)
@oracle_impl(
    hardware="B200",
    point="d9d8b8eb",
    STATS_BLOCK_X=16,
    STATS_BLOCK_R=128,
    FINAL_BLOCK_C=32,
    FINAL_BLOCK_R=64,
    CENTER_BLOCK_X=128,
    CENTER_BLOCK_R=64,
    stats_warps=8,
    final_warps=4,
    center_warps=8,
)
def oracle_forward(
    inputs,
    *,
    STATS_BLOCK_X: int,
    STATS_BLOCK_R: int,
    FINAL_BLOCK_C: int,
    FINAL_BLOCK_R: int,
    CENTER_BLOCK_X: int,
    CENTER_BLOCK_R: int,
    stats_warps: int,
    final_warps: int,
    center_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    partial_mean = torch.empty_strided(PARTIAL_SHAPE, PARTIAL_STRIDE, device=arg0_1.device, dtype=torch.float32)
    partial_m2 = torch.empty_strided(PARTIAL_SHAPE, PARTIAL_STRIDE, device=arg0_1.device, dtype=torch.float32)
    mean = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=arg0_1.device, dtype=torch.float32)
    pooled = torch.empty_strided(POOLED_SHAPE, POOLED_STRIDE, device=arg0_1.device, dtype=torch.bfloat16)
    centered = torch.empty_strided(INPUT_SHAPE, INPUT_STRIDE, device=arg0_1.device, dtype=torch.float32)

    _partial_stats_kernel[(triton.cdiv(C * HW, STATS_BLOCK_X),)](
        arg0_1,
        arg1_1,
        partial_mean,
        partial_m2,
        BLOCK_X=STATS_BLOCK_X,
        BLOCK_R=STATS_BLOCK_R,
        num_warps=stats_warps,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_mean,
        partial_m2,
        arg2_1,
        arg3_1,
        mean,
        invstd,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_R=FINAL_BLOCK_R,
        num_warps=final_warps,
        num_stages=3,
    )
    _center_pool_kernel[(triton.cdiv(N * C, CENTER_BLOCK_X),)](
        arg0_1,
        arg1_1,
        mean,
        invstd,
        arg4_1,
        arg5_1,
        pooled,
        centered,
        BLOCK_X=CENTER_BLOCK_X,
        BLOCK_R=CENTER_BLOCK_R,
        num_warps=center_warps,
        num_stages=3,
    )
    return invstd, pooled, centered, arg2_1, arg3_1
