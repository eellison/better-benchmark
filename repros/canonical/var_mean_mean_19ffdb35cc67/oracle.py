"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 ShuffleNetV2 training-BatchNorm ReLU spatial-mean scope with Triton kernels, including bf16-to-fp32 population `var_mean(..., correction=0)` over N/H/W, eps=1e-5 rsqrt, saved mean and rsqrt side outputs, mutable running-stat `copy_` aliases with the captured correction literal, fp32 affine math, the required bf16 round trip before ReLU, and the final bf16 `[N,C]` spatial mean, whereas Inductor lowers the BN statistics/update and downstream affine/ReLU/mean consumer through generic normalization and reduction schedules; Inductor cannot do this today because its BN-training template does not keep saved-stat outputs, mutable running-stat side effects, the bf16 cast boundary, and the immediate spatial-mean consumer in one full-scope channel plan; the fix is SCHEDULER_FUSION: extend the BN-training scheduler to emit saved stats and running-stat aliases while fusing fixed-shape affine bf16-ReLU plus spatial-mean stores into the normalization lowering."""

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
def _bn_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    saved_mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    channel = tl.program_id(0)
    e_offsets = tl.arange(0, BLOCK_E)
    hw = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    mask = e_offsets < E
    offsets = (
        n_idx * stride_n
        + channel * stride_c
        + h_idx * stride_h
        + w_idx * stride_w
    )

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    active = tl.where(mask, x, 0.0)
    sum_x = tl.sum(active, axis=0)
    sum_x2 = tl.sum(_f32_mul(active, active), axis=0)
    mean = _f32_div(sum_x, E + 0.0)
    ex2 = _f32_div(sum_x2, E + 0.0)
    var = _f32_sub(ex2, _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, 1.0001594642002871)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))

    tl.store(saved_mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)


@triton.jit
def _relu_spatial_mean_kernel(
    x_ptr,
    saved_mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    spatial_mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
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
    offsets = (
        n_idx[:, None] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(saved_mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    normalized = _f32_mul(_f32_sub(x, mean[:, None]), invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    zero = tl.full((ROW_BLOCK, BLOCK_HW), 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where(rounded != rounded, rounded, tl.maximum(rounded, zero)).to(tl.bfloat16)
    reduced = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1)
    mean_value = _f32_div(reduced, H * W + 0.0).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    tl.store(spatial_mean_ptr + rows, mean_value, mask=row_mask)


@oracle_impl(hardware="B200", point="5b9efb9c", BLOCK_E=8192, BLOCK_HW=64, ROW_BLOCK=8, STAT_WARPS=4, OUT_WARPS=1)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
    STAT_WARPS: int,
    OUT_WARPS: int,
):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    elements_per_channel = n * h * w
    total_rows = n * c

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
        (n, c),
        (c, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _bn_stats_kernel[(c,)](
        x,
        running_mean,
        running_var,
        saved_mean,
        invstd,
        C=c,
        H=h,
        W=w,
        E=elements_per_channel,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_E=BLOCK_E,
        num_warps=STAT_WARPS,
        num_stages=3,
    )
    _relu_spatial_mean_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        saved_mean,
        invstd,
        weight,
        bias,
        spatial_mean,
        C=c,
        H=h,
        W=w,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        TOTAL_ROWS=total_rows,
        BLOCK_HW=BLOCK_HW,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=3,
    )
    return saved_mean, invstd, spatial_mean, running_mean, running_var
