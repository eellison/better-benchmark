"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 training-BatchNorm ReLU spatial-mean scope, including bf16-to-fp32 population var_mean with correction=0, eps=0.001 rsqrt, saved mean and invstd side outputs, mutable running-stat copy_ returns with the captured 0.01 momentum and variance correction, explicit bf16 affine rounding before ReLU, returned bf16 activation, and the bf16 spatial mean reduced from that rounded ReLU, whereas Inductor lowers the BN statistics, mutable running-stat epilogue, activation materialization, and downstream spatial mean as generic separated regions; Inductor cannot do this today because its norm-template scheduler does not keep BN-training side outputs, side-effecting running-stat updates, visible activation stores, and immediate spatial-reduction consumers inside one full-scope channel-reduction plan; the fix is SCHEDULER_FUSION: extend the BN-training scheduler to emit saved stats and running-stat epilogues while sinking bf16 affine ReLU and spatial-mean consumers into the same planned lowering."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _partial_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n_block = tl.program_id(0)
    channel = tl.program_id(1)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    hw_offsets = tl.arange(0, BLOCK_HW)
    mask = (n_offsets[:, None] < N) & (hw_offsets[None, :] < HW)
    x_offsets = n_offsets[:, None] * C * HW + channel * HW + hw_offsets[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(x, axis=1)
    row_sumsq = tl.sum(x * x, axis=1)
    partial_offset = n_block * C + channel
    tl.store(partial_sum_ptr + partial_offset, tl.sum(row_sum, axis=0))
    tl.store(partial_sumsq_ptr + partial_offset, tl.sum(row_sumsq, axis=0))


@triton.jit
def _finalize_stats_kernel(
    partial_sum_ptr,
    partial_sumsq_ptr,
    running_mean_ptr,
    running_var_ptr,
    saved_mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    channel = tl.program_id(0)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    mask = split_offsets < NUM_SPLITS
    partial_offsets = split_offsets * C + channel
    sums = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
    sumsqs = tl.load(partial_sumsq_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(sums, axis=0)
    total_sumsq = tl.sum(sumsqs, axis=0)
    mean = total / E
    var = total_sumsq / E - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + 0.001)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(saved_mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, old_mean * 0.99 + mean * 0.01)
    tl.store(
        running_var_ptr + channel,
        old_var * 0.99 + var * RUNNING_VAR_CORRECTION * 0.01,
    )


@triton.jit
def _relu_spatial_mean_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    saved_mean_ptr,
    invstd_ptr,
    relu_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    hw_offsets = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < HW
    mask = row_mask[:, None] & hw_mask[None, :]
    channels = rows % C
    n_offsets = rows // C
    x_offsets = n_offsets[:, None] * C * HW + channels[:, None] * HW + hw_offsets[None, :]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(saved_mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    y = ((x - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
    y_bf16 = y.to(tl.bfloat16)
    relu = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    tl.store(relu_ptr + x_offsets, relu, mask=mask)

    pooled = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1) / HW
    tl.store(spatial_mean_ptr + rows, pooled.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="00cca0ad", BLOCK_N=8, BLOCK_HW=1024, ROW_BLOCK=8, stats_warps=8, final_warps=1, out_warps=4)
@oracle_impl(hardware="B200", point="b7fea7d6", BLOCK_N=8, BLOCK_HW=1024, ROW_BLOCK=8, stats_warps=8, final_warps=1, out_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
    stats_warps: int,
    final_warps: int,
    out_warps: int,
):
    x, running_mean, running_var, weight, bias = inputs
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    e = n * hw
    total_rows = n * c
    running_var_correction = e / (e - 1.0)
    num_splits = triton.cdiv(n, BLOCK_N)
    block_splits = triton.next_power_of_2(num_splits)

    saved_mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    relu = torch.empty_strided((n, c, h, w), (c * hw, hw, w, 1), device=x.device, dtype=torch.bfloat16)
    spatial_mean = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.bfloat16)
    partial_sum = torch.empty((num_splits, c), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty((num_splits, c), device=x.device, dtype=torch.float32)

    _partial_stats_kernel[(num_splits, c)](
        x,
        partial_sum,
        partial_sumsq,
        C=c,
        HW=hw,
        N=n,
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        num_warps=stats_warps,
        num_stages=3,
    )
    _finalize_stats_kernel[(c,)](
        partial_sum,
        partial_sumsq,
        running_mean,
        running_var,
        saved_mean,
        invstd,
        C=c,
        E=e,
        RUNNING_VAR_CORRECTION=running_var_correction,
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=block_splits,
        num_warps=final_warps,
        num_stages=1,
    )
    _relu_spatial_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        weight,
        bias,
        saved_mean,
        invstd,
        relu,
        spatial_mean,
        C=c,
        HW=hw,
        TOTAL_ROWS=total_rows,
        BLOCK_HW=BLOCK_HW,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=out_warps,
        num_stages=1,
    )
    return saved_mean, invstd, relu, spatial_mean, running_mean, running_var
