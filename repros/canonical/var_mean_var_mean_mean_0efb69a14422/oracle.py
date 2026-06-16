"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG bf16 dual training-BatchNorm scope, including both bf16-to-fp32 population var_mean reductions, eps=1e-5 rsqrt side outputs, four mutable running-stat copy_ updates with the captured correction literal, bf16 affine rounding on each branch, bf16 branch add, NaN-preserving ReLU, and final bf16 7x7 spatial mean view, whereas Inductor lowers the sibling BN-training reductions and downstream activation/spatial mean through separate generic normalization and reduction schedules; Inductor cannot do this today because normalization template canonicalization does not fuse paired training-BN statistics, mutable running-stat returns, observable bf16 cast boundaries, and the shared post-add spatial reduction into one full-scope channel schedule; the fix is SCHEDULER_FUSION: teach the scheduler/norm template to keep paired per-channel BN statistics, running-stat updates, bf16 branch epilogues, and the post-sum spatial mean in one fused schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _dual_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    S0N: tl.constexpr,
    S0C: tl.constexpr,
    S0H: tl.constexpr,
    S0W: tl.constexpr,
    S1N: tl.constexpr,
    S1C: tl.constexpr,
    S1H: tl.constexpr,
    S1W: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    hw_size: tl.constexpr = H * W
    hw = e_offsets % hw_size
    n_idx = e_offsets // hw_size
    h_idx = hw // W
    w_idx = hw - h_idx * W
    mask = (c_offsets[:, None] < C) & (e_offsets[None, :] < E)

    off0 = (
        n_idx[None, :] * S0N
        + c_offsets[:, None] * S0C
        + h_idx[None, :] * S0H
        + w_idx[None, :] * S0W
    )
    off1 = (
        n_idx[None, :] * S1N
        + c_offsets[:, None] * S1C
        + h_idx[None, :] * S1H
        + w_idx[None, :] * S1W
    )
    vals0 = tl.load(x0_ptr + off0, mask=mask, other=0.0).to(tl.float32)
    vals1 = tl.load(x1_ptr + off1, mask=mask, other=0.0).to(tl.float32)
    active0 = tl.where(mask, vals0, 0.0)
    active1 = tl.where(mask, vals1, 0.0)
    sum0 = tl.sum(active0, axis=1)
    sum_sq0 = tl.sum(_f32_mul(active0, active0), axis=1)
    sum1 = tl.sum(active1, axis=1)
    sum_sq1 = tl.sum(_f32_mul(active1, active1), axis=1)

    chunk = tl.program_id(1)
    out_offsets = chunk * C + c_offsets
    plane = NUM_CHUNKS * C
    c_mask = c_offsets < C
    tl.store(partial_mean_ptr + out_offsets, sum0, mask=c_mask)
    tl.store(partial_m2_ptr + out_offsets, sum_sq0, mask=c_mask)
    tl.store(partial_mean_ptr + plane + out_offsets, sum1, mask=c_mask)
    tl.store(partial_m2_ptr + plane + out_offsets, sum_sq1, mask=c_mask)


@triton.jit
def _dual_finalize_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    saved_mean0_ptr,
    invstd0_ptr,
    saved_mean1_ptr,
    invstd1_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_offsets = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    branch = tl.program_id(1)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    c_mask = c_offsets < C
    mask = c_mask[:, None] & (chunks[None, :] < NUM_CHUNKS)
    plane = NUM_CHUNKS * C
    offsets = branch * plane + chunks[None, :] * C + c_offsets[:, None]
    sums = tl.load(partial_mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums_sq = tl.load(partial_m2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, sums, 0.0), axis=1)
    total_sq = tl.sum(tl.where(mask, sums_sq, 0.0), axis=1)
    mean = total / E
    ex2 = total_sq / E
    var = tl.maximum(_f32_sub(ex2, _f32_mul(mean, mean)), 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=(branch == 0) & c_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + c_offsets, mask=(branch == 0) & c_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=(branch == 1) & c_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + c_offsets, mask=(branch == 1) & c_mask, other=0.0).to(tl.float32)

    mean_update = _f32_mul(mean, 0.1)
    var_update = _f32_mul(_f32_mul(var, 1.0001594642002871), 0.1)
    new_mean0 = _f32_add(_f32_mul(old_mean0, 0.9), mean_update)
    new_var0 = _f32_add(_f32_mul(old_var0, 0.9), var_update)
    new_mean1 = _f32_add(_f32_mul(old_mean1, 0.9), mean_update)
    new_var1 = _f32_add(_f32_mul(old_var1, 0.9), var_update)

    tl.store(running_mean0_ptr + c_offsets, new_mean0, mask=(branch == 0) & c_mask)
    tl.store(running_var0_ptr + c_offsets, new_var0, mask=(branch == 0) & c_mask)
    tl.store(saved_mean0_ptr + c_offsets, mean, mask=(branch == 0) & c_mask)
    tl.store(invstd0_ptr + c_offsets, invstd, mask=(branch == 0) & c_mask)
    tl.store(running_mean1_ptr + c_offsets, new_mean1, mask=(branch == 1) & c_mask)
    tl.store(running_var1_ptr + c_offsets, new_var1, mask=(branch == 1) & c_mask)
    tl.store(saved_mean1_ptr + c_offsets, mean, mask=(branch == 1) & c_mask)
    tl.store(invstd1_ptr + c_offsets, invstd, mask=(branch == 1) & c_mask)


@triton.jit
def _dual_bn_relu_spatial_mean_kernel(
    x0_ptr,
    weight0_ptr,
    bias0_ptr,
    x1_ptr,
    weight1_ptr,
    bias1_ptr,
    saved_mean0_ptr,
    invstd0_ptr,
    saved_mean1_ptr,
    invstd1_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S0N: tl.constexpr,
    S0C: tl.constexpr,
    S0H: tl.constexpr,
    S0W: tl.constexpr,
    S1N: tl.constexpr,
    S1C: tl.constexpr,
    S1H: tl.constexpr,
    S1W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    n_idx = tl.program_id(0)
    c_offsets = tl.program_id(1) * C_BLOCK + tl.arange(0, C_BLOCK)
    hw_offsets = tl.arange(0, BLOCK_HW)
    hw_size: tl.constexpr = H * W
    h_idx = hw_offsets // W
    w_idx = hw_offsets - h_idx * W
    valid = (c_offsets[:, None] < C) & (hw_offsets[None, :] < hw_size)

    off0 = (
        n_idx * S0N
        + c_offsets[:, None] * S0C
        + h_idx[None, :] * S0H
        + w_idx[None, :] * S0W
    )
    off1 = (
        n_idx * S1N
        + c_offsets[:, None] * S1C
        + h_idx[None, :] * S1H
        + w_idx[None, :] * S1W
    )
    x0 = tl.load(x0_ptr + off0, mask=valid, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + off1, mask=valid, other=0.0).to(tl.float32)

    c_mask = c_offsets < C
    mean0 = tl.load(saved_mean0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd0 = tl.load(invstd0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    mean1 = tl.load(saved_mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd1 = tl.load(invstd1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    y0 = _f32_add(
        _f32_mul(_f32_mul(_f32_sub(x0, mean0[:, None]), invstd0[:, None]), weight0[:, None]),
        bias0[:, None],
    ).to(tl.bfloat16)
    y1 = _f32_add(
        _f32_mul(_f32_mul(_f32_sub(x1, mean1[:, None]), invstd1[:, None]), weight1[:, None]),
        bias1[:, None],
    ).to(tl.bfloat16)
    added = (y0 + y1).to(tl.bfloat16)
    zero = tl.full((C_BLOCK, BLOCK_HW), 0.0, tl.bfloat16)
    relu = tl.where(added < zero, zero, added)
    pooled = tl.sum(tl.where(valid, relu.to(tl.float32), 0.0), axis=1) * (1.0 / hw_size)
    tl.store(out_ptr + n_idx * C + c_offsets, pooled.to(tl.bfloat16), mask=c_mask)


# e1dd88f2: dual RepVGG BN training, bf16 channels-last inputs, bf16 [128,1408] spatial mean.
@oracle_impl(hardware="B200", point="e1dd88f2")
def oracle_forward(inputs):
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    n = x0.shape[0]
    c = x0.shape[1]
    h = x0.shape[2]
    w = x0.shape[3]
    e = n * h * w
    block_e = 1024
    c_block = 8
    out_c_block = 16
    num_chunks = triton.cdiv(e, block_e)
    block_chunks = _next_power_of_2(num_chunks)

    saved_mean0 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd0 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    saved_mean1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x0.device, dtype=torch.float32)
    spatial_mean = torch.empty_strided((n, c), (c, 1), device=x0.device, dtype=torch.bfloat16)
    partial_mean = torch.empty((2, num_chunks, c), device=x0.device, dtype=torch.float32)
    partial_m2 = torch.empty((2, num_chunks, c), device=x0.device, dtype=torch.float32)

    _dual_partial_stats_kernel[(triton.cdiv(c, c_block), num_chunks)](
        x0,
        x1,
        partial_mean,
        partial_m2,
        c,
        h,
        w,
        e,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x1.stride(0),
        x1.stride(1),
        x1.stride(2),
        x1.stride(3),
        num_chunks,
        BLOCK_E=block_e,
        C_BLOCK=c_block,
        num_warps=8,
        num_stages=3,
    )
    _dual_finalize_stats_kernel[(triton.cdiv(c, c_block), 2)](
        partial_mean,
        partial_m2,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        saved_mean0,
        invstd0,
        saved_mean1,
        invstd1,
        c,
        e,
        num_chunks,
        BLOCK_E=block_e,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=c_block,
        num_warps=1,
        num_stages=3,
    )
    _dual_bn_relu_spatial_mean_kernel[(n, triton.cdiv(c, out_c_block))](
        x0,
        weight0,
        bias0,
        x1,
        weight1,
        bias1,
        saved_mean0,
        invstd0,
        saved_mean1,
        invstd1,
        spatial_mean,
        c,
        h,
        w,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x1.stride(0),
        x1.stride(1),
        x1.stride(2),
        x1.stride(3),
        BLOCK_HW=triton.next_power_of_2(h * w),
        C_BLOCK=out_c_block,
        num_warps=1,
        num_stages=3,
    )
    return (
        saved_mean0,
        invstd0,
        saved_mean1,
        invstd1,
        spatial_mean,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )
