"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 UNet training-BatchNorm bilinear-upsample-cat scope for the captured `[1, 512, 40, 59]` point, including bf16-to-fp32 population `var_mean(..., correction=0)`, eps=1e-5 `rsqrt`, returned mean/rsqrt tensors, mutable running-stat `copy_` aliases with the captured variance correction, exact bf16 affine/ReLU cast boundary, all returned interpolation index and weight tensors, and direct bf16 `[1, 1024, 80, 119]` cat output stores, whereas Inductor lowers the normalization, running-stat epilogues, dtype-boundary activation, bilinear unsafe-index interpolation, padding, and cat as separate generic regions; Inductor cannot do this today because its training-normalization scheduler does not sink fixed bilinear upsample/pad/cat consumers and visible side tensors into one dtype-aware BN epilogue; the fix is SCHEDULER_FUSION: extend the BN-training template to emit saved stats, running-stat aliases, interpolation metadata, and direct static upsample-cat stores while preserving bf16 rounding boundaries."""

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
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _bn_relu_value(x_ptr, mean, invstd, weight, bias, offsets, mask):
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(norm, weight), bias)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    return _relu_preserve_nan(rounded)


@triton.jit
def _stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_R)
    mask = offsets < E
    x_offsets = channel * H * W + offsets
    vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    vals = tl.where(mask, vals, 0.0)

    total = tl.sum(vals, axis=0)
    total2 = tl.sum(_f32_mul(vals, vals), axis=0)
    mean = total / E
    var = _f32_sub(total2 / E, _f32_mul(mean, mean))
    var = tl.maximum(var, 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, RUNNING_VAR_CORRECTION)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))

    tl.store(mean_ptr + channel, mean, mask=channel < C)
    tl.store(invstd_ptr + channel, invstd, mask=channel < C)
    tl.store(running_mean_ptr + channel, new_mean, mask=channel < C)
    tl.store(running_var_ptr + channel, new_var, mask=channel < C)


@triton.jit
def _indices_kernel(
    row0_ptr,
    row1_ptr,
    col0_ptr,
    col1_ptr,
    col_alpha_ptr,
    row_alpha_ptr,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)

    row_mask = idx < 80
    row_f = idx.to(tl.float32) * 0.4936708860759494
    row0 = row_f.to(tl.int64)
    row1 = tl.minimum(row0 + 1, 39)
    row_delta = _f32_sub(row_f, row0.to(tl.float32))
    row_delta = tl.maximum(tl.minimum(row_delta, 1.0), 0.0)
    tl.store(row0_ptr + idx, row0, mask=row_mask)
    tl.store(row1_ptr + idx, row1, mask=row_mask)
    tl.store(row_alpha_ptr + idx, row_delta, mask=row_mask)

    col_mask = idx < 118
    col_f = idx.to(tl.float32) * 0.49572649572649574
    col0 = col_f.to(tl.int64)
    col1 = tl.minimum(col0 + 1, 58)
    col_delta = _f32_sub(col_f, col0.to(tl.float32))
    col_delta = tl.maximum(tl.minimum(col_delta, 1.0), 0.0)
    tl.store(col0_ptr + idx, col0, mask=col_mask)
    tl.store(col1_ptr + idx, col1, mask=col_mask)
    tl.store(col_alpha_ptr + idx, col_delta, mask=col_mask)


@triton.jit
def _upsample_cat_kernel(
    x_ptr,
    skip_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    row0_ptr,
    row1_ptr,
    col0_ptr,
    col1_ptr,
    col_alpha_ptr,
    row_alpha_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    UPSAMPLE_W: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < TOTAL

    ow = offsets % OUT_W
    t0 = offsets // OUT_W
    oh = t0 % OUT_H
    channel = (t0 // OUT_H) % (C + C)
    first_half = channel < C
    branch_channel = channel - C

    skip_offsets = channel * OUT_H * OUT_W + oh * OUT_W + ow
    skip_value = tl.load(skip_ptr + skip_offsets, mask=mask & first_half, other=0.0)

    valid_branch = mask & (~first_half) & (ow < UPSAMPLE_W)
    h0 = tl.load(row0_ptr + oh, mask=valid_branch, other=0).to(tl.int64)
    h1 = tl.load(row1_ptr + oh, mask=valid_branch, other=0).to(tl.int64)
    w0 = tl.load(col0_ptr + ow, mask=valid_branch, other=0).to(tl.int64)
    w1 = tl.load(col1_ptr + ow, mask=valid_branch, other=0).to(tl.int64)
    row_alpha = tl.load(row_alpha_ptr + oh, mask=valid_branch, other=0.0).to(tl.float32)
    col_alpha = tl.load(col_alpha_ptr + ow, mask=valid_branch, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + branch_channel, mask=valid_branch, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + branch_channel, mask=valid_branch, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + branch_channel, mask=valid_branch, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + branch_channel, mask=valid_branch, other=0.0).to(tl.float32)

    input_base = branch_channel * IN_H * IN_W
    v00 = _bn_relu_value(x_ptr, mean, invstd, weight, bias, input_base + h0 * IN_W + w0, valid_branch)
    v01 = _bn_relu_value(x_ptr, mean, invstd, weight, bias, input_base + h0 * IN_W + w1, valid_branch)
    v10 = _bn_relu_value(x_ptr, mean, invstd, weight, bias, input_base + h1 * IN_W + w0, valid_branch)
    v11 = _bn_relu_value(x_ptr, mean, invstd, weight, bias, input_base + h1 * IN_W + w1, valid_branch)

    top = _f32_add(v00, _f32_mul(_f32_sub(v01, v00), col_alpha))
    bottom = _f32_add(v10, _f32_mul(_f32_sub(v11, v10), col_alpha))
    interp = _f32_add(top, _f32_mul(_f32_sub(bottom, top), row_alpha))
    branch_value = tl.where(ow < UPSAMPLE_W, interp.to(tl.bfloat16, fp_downcast_rounding="rtne"), 0.0)
    value = tl.where(first_half, skip_value, branch_value)
    tl.store(out_ptr + offsets, value, mask=mask)


# (T([1,512,40,59], bf16), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([1,512,80,119], bf16), S([80,1]))
@oracle_impl(
    hardware="B200",
    point="7ff155f1",
    BLOCK_R=4096,
    INDEX_BLOCK=128,
    OUT_BLOCK=256,
    STATS_WARPS=8,
    OUT_WARPS=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    INDEX_BLOCK: int,
    OUT_BLOCK: int,
    STATS_WARPS: int,
    OUT_WARPS: int,
    num_stages: int,
):
    x, running_mean, running_var, weight, bias, skip, _shape_param_0 = inputs
    channels = 512
    in_h = 40
    in_w = 59
    out_h = 80
    out_w = 119
    upsample_w = 118
    elems_per_channel = in_h * in_w
    total_out = (channels + channels) * out_h * out_w

    mean = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    row0 = torch.empty_strided((out_h, 1), (1, 1), device=x.device, dtype=torch.int64)
    row1 = torch.empty_strided((out_h, 1), (1, 1), device=x.device, dtype=torch.int64)
    col0 = torch.empty_strided((upsample_w,), (1,), device=x.device, dtype=torch.int64)
    col1 = torch.empty_strided((upsample_w,), (1,), device=x.device, dtype=torch.int64)
    col_alpha = torch.empty_strided((upsample_w,), (1,), device=x.device, dtype=torch.float32)
    row_alpha = torch.empty_strided((out_h, 1), (1, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (1, channels + channels, out_h, out_w),
        ((channels + channels) * out_h * out_w, out_h * out_w, out_w, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _stats_kernel[(channels,)](
        x,
        running_mean,
        running_var,
        mean,
        invstd,
        C=channels,
        H=in_h,
        W=in_w,
        E=elems_per_channel,
        RUNNING_VAR_CORRECTION=1.0004239084357778,
        BLOCK_R=BLOCK_R,
        num_warps=STATS_WARPS,
        num_stages=num_stages,
    )
    _indices_kernel[(1,)](
        row0,
        row1,
        col0,
        col1,
        col_alpha,
        row_alpha,
        BLOCK=INDEX_BLOCK,
        num_warps=4,
        num_stages=num_stages,
    )
    _upsample_cat_kernel[(triton.cdiv(total_out, OUT_BLOCK),)](
        x,
        skip,
        weight,
        bias,
        mean,
        invstd,
        row0,
        row1,
        col0,
        col1,
        col_alpha,
        row_alpha,
        out,
        TOTAL=total_out,
        C=channels,
        IN_H=in_h,
        IN_W=in_w,
        OUT_H=out_h,
        OUT_W=out_w,
        UPSAMPLE_W=upsample_w,
        BLOCK_E=OUT_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=num_stages,
    )
    return mean, invstd, row0, row1, col0, col1, col_alpha, row_alpha, out, running_mean, running_var
