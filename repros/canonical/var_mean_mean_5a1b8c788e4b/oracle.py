"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 training-BatchNorm ReLU plus spatial-mean scope with split channel statistics and a fused activation/mean epilogue, including fp32 population `var_mean(..., correction=0)` over the bf16 channels-last input, eps=1e-5 rsqrt, saved mean and rsqrt side outputs, mutable running-stat `copy_` outputs with the captured correction literal, fp32 affine, explicit bf16 rounding before ReLU, the returned full bf16 activation, and the returned bf16 spatial mean, whereas Inductor lowers the canonicalized var_mean/update/affine/cast/ReLU/mean graph through generic normalization and reduction schedules; Inductor cannot do this today because its norm-template scheduler does not keep mutable BN side outputs, the visible bf16 activation materialization, and the downstream spatial reduction in one full-scope channels-last plan; the fix is SCHEDULER_FUSION: extend the BN-training template to emit saved statistics and running-stat side effects while sinking fixed-shape affine ReLU plus spatial-mean consumers into the channel-statistics lowering."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _partial_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    r_block = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    hw = r_offsets % (H * W)
    n_idx = r_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (r_offsets[None, :] < E)
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(values, axis=1)
    sums2 = tl.sum(_f32_mul(values, values), axis=1)
    out_offsets = r_block * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channels < C)


@triton.jit
def _finalize_stats_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean_ptr,
    running_var_ptr,
    saved_mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(sums, axis=1)
    total2 = tl.sum(sums2, axis=1)

    mean = total / E
    var = _f32_sub(total2 / E, _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(old_mean, 0.9), _f32_mul(mean, 0.1))
    corrected_var = _f32_mul(var, 1.0000024912370735)
    new_var = _f32_add(_f32_mul(old_var, 0.9), _f32_mul(corrected_var, 0.1))

    tl.store(saved_mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, new_var, mask=channel_mask)


@triton.jit
def _relu_spatial_mean_kernel(
    x_ptr,
    saved_mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    activation_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    hw_offsets = tl.arange(0, BLOCK_HW)
    channels = rows - (rows // C) * C
    n_idx = rows // C
    h_idx = hw_offsets // W
    w_idx = hw_offsets - h_idx * W
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < (H * W)
    mask = row_mask[:, None] & hw_mask[None, :]
    x_offsets = (
        n_idx[:, None] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    out_offsets = (
        n_idx[:, None] * out_stride_n
        + channels[:, None] * out_stride_c
        + h_idx[None, :] * out_stride_h
        + w_idx[None, :] * out_stride_w
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(saved_mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    normalized = _f32_mul(_f32_sub(x, mean[:, None]), invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    rounded = affine.to(tl.bfloat16)
    relu = tl.where(rounded != rounded, rounded, tl.maximum(rounded, 0.0)).to(tl.bfloat16)

    tl.store(activation_ptr + out_offsets, relu, mask=mask)
    reduced = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1)
    mean_value = (reduced / (H * W)).to(tl.bfloat16)
    tl.store(spatial_mean_ptr + rows, mean_value, mask=row_mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="765e4345", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=1024, ROW_BLOCK=8, STAT_WARPS=8, FINAL_WARPS=1, OUT_WARPS=8)
@oracle_impl(hardware="B200", point="0cb3f08a", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=1024, ROW_BLOCK=8, STAT_WARPS=8, FINAL_WARPS=1, OUT_WARPS=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
    STAT_WARPS: int,
    FINAL_WARPS: int,
    OUT_WARPS: int,
):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    elements_per_channel = n * h * w
    total_rows = n * c
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    saved_mean = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    activation = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    spatial_mean = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty_strided(
        (num_chunks, c),
        (c, 1),
        device=x.device,
        dtype=torch.float32,
    )
    partial_sum2 = torch.empty_strided(
        (num_chunks, c),
        (c, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _partial_stats_kernel[(triton.cdiv(c, BLOCK_C), num_chunks)](
        x,
        partial_sum,
        partial_sum2,
        C=c,
        H=h,
        W=w,
        E=elements_per_channel,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=STAT_WARPS,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(c, BLOCK_C),)](
        partial_sum,
        partial_sum2,
        running_mean,
        running_var,
        saved_mean,
        invstd,
        C=c,
        E=elements_per_channel,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=BLOCK_C,
        num_warps=FINAL_WARPS,
        num_stages=3,
    )
    _relu_spatial_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        saved_mean,
        invstd,
        weight,
        bias,
        activation,
        spatial_mean,
        C=c,
        H=h,
        W=w,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        out_stride_n=int(activation.stride(0)),
        out_stride_c=int(activation.stride(1)),
        out_stride_h=int(activation.stride(2)),
        out_stride_w=int(activation.stride(3)),
        TOTAL_ROWS=total_rows,
        BLOCK_HW=BLOCK_HW,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=3,
    )
    return saved_mean, invstd, activation, spatial_mean, running_mean, running_var
