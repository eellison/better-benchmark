"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT/EfficientNet bf16 training-BatchNorm SiLU spatial-mean scope with split channel statistics and a fused finalize/activation/pool epilogue, including bf16-to-fp32 population var_mean over N/H/W, eps=1e-5 rsqrt, saved mean and rsqrt side outputs, mutable running-stat copy_ aliases with the captured variance-correction literal, fp32 affine, the required bf16 round trip before natural-exp SiLU, bf16 SiLU before the final spatial mean, and the returned [N,C] view, whereas Inductor lowers the BN statistics/update, affine/SiLU cast boundary, and downstream spatial mean through separate generic normalization and reduction schedules; Inductor cannot do this today because the normalization scheduler does not keep training-BN side outputs, mutable stat epilogues, explicit bf16 activation boundaries, and the immediate spatial-mean consumer in one full-scope channel plan; the fix is SCHEDULER_FUSION: extend the BN-training template to emit saved statistics and running-stat aliases while fusing affine, natural-exp SiLU, and fixed-spatial mean stores from the same schedule."""

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
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    channels = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    hw: tl.constexpr = H * W
    n_idx = e_offsets // hw
    spatial = e_offsets - n_idx * hw
    h_idx = spatial // W
    w_idx = spatial - h_idx * W
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    active = tl.where(mask, values, 0.0)
    sums = tl.sum(active, axis=1)
    sums2 = tl.sum(_f32_mul(active, active), axis=1)
    out_offsets = tl.program_id(1) * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channels < C)


@triton.jit
def _finalize_silu_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
    saved_mean_ptr,
    invstd_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    channels = rows - (rows // C) * C
    n_idx = rows // C
    row_mask = rows < TOTAL_ROWS
    partial_mask = (chunks[None, :] < NUM_CHUNKS) & row_mask[:, None]
    partial_offsets = chunks[None, :] * C + channels[:, None]
    sums = tl.load(partial_sum_ptr + partial_offsets, mask=partial_mask, other=0.0).to(tl.float32)
    sums2 = tl.load(partial_sum2_ptr + partial_offsets, mask=partial_mask, other=0.0).to(tl.float32)
    total = tl.sum(sums, axis=1)
    total2 = tl.sum(sums2, axis=1)

    mean = total / E
    ex2 = total2 / E
    var = _f32_sub(ex2, _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    update_mask = row_mask & (n_idx == 0)
    old_mean = tl.load(running_mean_ptr + channels, mask=update_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=update_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(mean * 0.1, old_mean * 0.9)
    corrected_var = var * RUNNING_VAR_CORRECTION
    new_var = _f32_add(corrected_var * 0.1, old_var * 0.9)

    tl.store(saved_mean_ptr + channels, mean, mask=update_mask)
    tl.store(invstd_ptr + channels, invstd, mask=update_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=update_mask)
    tl.store(running_var_ptr + channels, new_var, mask=update_mask)

    hw_offsets = tl.arange(0, BLOCK_HW)
    hw: tl.constexpr = H * W
    hw_mask = hw_offsets < hw
    h_idx = hw_offsets // W
    w_idx = hw_offsets - h_idx * W
    elem_mask = row_mask[:, None] & hw_mask[None, :]
    x_offsets = (
        n_idx[:, None] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    x = tl.load(x_ptr + x_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    centered = x - mean[:, None]
    normalized = centered * invstd[:, None]
    scaled = normalized * weight[:, None]
    affine = scaled + bias[:, None]
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    neg = -rounded
    denom = libdevice.exp(neg) + 1.0
    silu = (rounded / denom).to(tl.bfloat16, fp_downcast_rounding="rtne")
    pooled = tl.sum(tl.where(elem_mask, silu.to(tl.float32), 0.0), axis=1)
    mean_value = (pooled / hw).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(spatial_mean_ptr + rows, mean_value, mask=row_mask)


@oracle_impl(
    hardware="B200",
    point="db4bc2f7",
    BLOCK_E=4096,
    C_BLOCK=16,
    ROW_BLOCK=64,
    STAT_WARPS=8,
    OUT_WARPS=4,
)
@oracle_impl(
    hardware="B200",
    point="86b98b1a",
    BLOCK_E=4096,
    C_BLOCK=32,
    ROW_BLOCK=32,
    STAT_WARPS=8,
    OUT_WARPS=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    ROW_BLOCK: int,
    STAT_WARPS: int,
    OUT_WARPS: int,
):
    x, running_mean, running_var, weight, bias, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    elements_per_channel = n * h * w
    total_rows = n * c
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_E)
    block_chunks = _next_power_of_2(num_chunks)
    running_var_correction = elements_per_channel / (elements_per_channel - 1.0)

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
    spatial_mean = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_0),
        tuple(int(dim) for dim in _shape_param_1),
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

    _partial_stats_kernel[(triton.cdiv(c, C_BLOCK), num_chunks)](
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
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=STAT_WARPS,
        num_stages=3,
    )
    _finalize_silu_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        partial_sum,
        partial_sum2,
        saved_mean,
        invstd,
        spatial_mean,
        C=c,
        H=h,
        W=w,
        E=elements_per_channel,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        TOTAL_ROWS=total_rows,
        RUNNING_VAR_CORRECTION=running_var_correction,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_HW=triton.next_power_of_2(h * w),
        num_warps=OUT_WARPS,
        num_stages=1,
    )
    return saved_mean, invstd, spatial_mean.view(tuple(int(dim) for dim in _shape_param_2)), running_mean, running_var
