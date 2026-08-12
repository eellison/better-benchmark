"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete pytorch_unet bf16 training-BatchNorm plus bilinear-upsampling cat scope, including bf16-to-fp32 channel statistics over `[N,H,W]`, returned mean and eps=1e-5 rsqrt side outputs, the exact running-mean/running-var `copy_` aliases with the captured unbiased-variance correction, affine bf16 rounding before ReLU, explicit align-corners index/weight tensors, right-padding of the upsampled branch, and the returned bf16 `[1,256,320,479]` channel cat. Inductor lowers the BN statistics, mutable running-stat epilogue, affine/ReLU producer, four indexed upsample gathers, pad, and cat through generic scheduler boundaries; the fix is SCHEDULER_FUSION: extend the training-BatchNorm/layout scheduler to preserve mutable side outputs while sinking fixed bilinear upsample, pad, and cat stores into one full-scope plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


RUNNING_VAR_CORRECTION = 1.0000261513114883
SCALE_H = 0.49843260188087773
SCALE_W = 0.4989517819706499


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
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    r_block = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    hw = r_offsets % (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = channels[:, None] * H * W + h_idx[None, :] * W + w_idx[None, :]
    mask = (channels[:, None] < C) & (r_offsets[None, :] < E)
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(values, axis=1)
    sums2 = tl.sum(_f32_mul(values, values), axis=1)
    out_offsets = r_block * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channels < C)


@triton.jit
def _finalize_stats_and_indices_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    y0_ptr,
    y1_ptr,
    x0_ptr,
    x1_ptr,
    x_weight_ptr,
    y_weight_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W_VALID: tl.constexpr,
    IN_H_MAX: tl.constexpr,
    IN_W_MAX: tl.constexpr,
    RUNNING_VAR_CORRECTION_C: tl.constexpr,
    SCALE_H_C: tl.constexpr,
    SCALE_W_C: tl.constexpr,
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
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, RUNNING_VAR_CORRECTION_C)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))

    tl.store(mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, new_var, mask=channel_mask)

    idx = channels
    y_mask = idx < OUT_H
    y_float = _f32_mul(idx.to(tl.float32), SCALE_H_C)
    y0 = y_float.to(tl.int64)
    y1 = tl.minimum(y0 + 1, IN_H_MAX)
    y_weight = tl.minimum(tl.maximum(_f32_sub(y_float, y0.to(tl.float32)), 0.0), 1.0)
    tl.store(y0_ptr + idx, y0, mask=y_mask)
    tl.store(y1_ptr + idx, y1, mask=y_mask)
    tl.store(y_weight_ptr + idx, y_weight, mask=y_mask)

    x_mask = idx < OUT_W_VALID
    x_float = _f32_mul(idx.to(tl.float32), SCALE_W_C)
    x0 = x_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, IN_W_MAX)
    x_weight = tl.minimum(tl.maximum(_f32_sub(x_float, x0.to(tl.float32)), 0.0), 1.0)
    tl.store(x0_ptr + idx, x0, mask=x_mask)
    tl.store(x1_ptr + idx, x1, mask=x_mask)
    tl.store(x_weight_ptr + idx, x_weight, mask=x_mask)


@triton.jit
def _relu_value(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    channels,
    h_idx,
    w_idx,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    mask,
):
    offsets = channels * H * W + h_idx * W + w_idx
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    rounded = affine.to(tl.bfloat16)
    relu = tl.where(rounded != rounded, rounded, tl.maximum(rounded, 0.0)).to(tl.bfloat16)
    return relu.to(tl.float32)


@triton.jit
def _cat_upsample_kernel(
    x_ptr,
    skip_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    cat_ptr,
    C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    TOTAL: tl.constexpr,
    SCALE_H_C: tl.constexpr,
    SCALE_W_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    w_out = offsets % OUT_W
    h_out = (offsets // OUT_W) % OUT_H
    c_out = (offsets // (OUT_H * OUT_W)) % (C * 2)
    first_half = c_out < C
    c = tl.where(first_half, c_out, c_out - C)

    skip_offsets = c * OUT_H * OUT_W + h_out * OUT_W + w_out
    skip_value = tl.load(skip_ptr + skip_offsets, mask=mask & first_half, other=0.0)

    valid_upsample = mask & ~first_half & (w_out < (OUT_W - 1))
    y_float = _f32_mul(h_out.to(tl.float32), SCALE_H_C)
    y0 = y_float.to(tl.int64)
    y1 = tl.minimum(y0 + 1, IN_H - 1)
    y_weight = tl.minimum(tl.maximum(_f32_sub(y_float, y0.to(tl.float32)), 0.0), 1.0)

    x_float = _f32_mul(w_out.to(tl.float32), SCALE_W_C)
    x0 = x_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, IN_W - 1)
    x_weight = tl.minimum(tl.maximum(_f32_sub(x_float, x0.to(tl.float32)), 0.0), 1.0)

    v00 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y0, x0, C, IN_H, IN_W, valid_upsample)
    v01 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y0, x1, C, IN_H, IN_W, valid_upsample)
    v10 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y1, x0, C, IN_H, IN_W, valid_upsample)
    v11 = _relu_value(x_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, c, y1, x1, C, IN_H, IN_W, valid_upsample)

    top = _f32_add(v00, _f32_mul(_f32_sub(v01, v00), x_weight))
    bottom = _f32_add(v10, _f32_mul(_f32_sub(v11, v10), x_weight))
    interp = _f32_add(top, _f32_mul(_f32_sub(bottom, top), y_weight))
    upsample_value = tl.where(w_out < (OUT_W - 1), interp.to(tl.bfloat16), 0.0)
    value = tl.where(first_half, skip_value, upsample_value)
    tl.store(cat_ptr + offsets, value, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# d2d98efc: (T([1,128,160,239], bf16), T([128], f32), T([128], f32), T([128], f32), T([128], f32), T([1,128,320,479], bf16), ...)
@oracle_impl(
    hardware="B200",
    point="d2d98efc",
    BLOCK_R=1024,
    BLOCK_C=8,
    FINAL_C_BLOCK=8,
    OUT_BLOCK=512,
    STAT_WARPS=4,
    FINAL_WARPS=1,
    OUT_WARPS=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_C_BLOCK: int,
    OUT_BLOCK: int,
    STAT_WARPS: int,
    FINAL_WARPS: int,
    OUT_WARPS: int,
):
    x, running_mean, running_var, weight, bias, skip, _shape0 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    out_h = int(skip.shape[2])
    out_w = int(skip.shape[3])
    elements_per_channel = n * h * w
    num_chunks = triton.cdiv(elements_per_channel, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    y0 = torch.empty_strided((out_h, 1), (1, 1), device=x.device, dtype=torch.int64)
    y1 = torch.empty_strided((out_h, 1), (1, 1), device=x.device, dtype=torch.int64)
    x0 = torch.empty_strided((out_w - 1,), (1,), device=x.device, dtype=torch.int64)
    x1 = torch.empty_strided((out_w - 1,), (1,), device=x.device, dtype=torch.int64)
    x_weight = torch.empty_strided((out_w - 1,), (1,), device=x.device, dtype=torch.float32)
    y_weight = torch.empty_strided((out_h, 1), (1, 1), device=x.device, dtype=torch.float32)
    cat = torch.empty_strided(
        (n, c * 2, out_h, out_w),
        _contiguous_stride((n, c * 2, out_h, out_w)),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty_strided((num_chunks, c), (c, 1), device=x.device, dtype=torch.float32)
    partial_sum2 = torch.empty_strided((num_chunks, c), (c, 1), device=x.device, dtype=torch.float32)

    _partial_stats_kernel[(triton.cdiv(c, BLOCK_C), num_chunks)](
        x,
        partial_sum,
        partial_sum2,
        C=c,
        H=h,
        W=w,
        E=elements_per_channel,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=STAT_WARPS,
        num_stages=3,
    )
    _finalize_stats_and_indices_kernel[(triton.cdiv(max(c, out_h, out_w - 1), FINAL_C_BLOCK),)](
        partial_sum,
        partial_sum2,
        running_mean,
        running_var,
        mean,
        invstd,
        y0,
        y1,
        x0,
        x1,
        x_weight,
        y_weight,
        C=c,
        E=elements_per_channel,
        OUT_H=out_h,
        OUT_W_VALID=out_w - 1,
        IN_H_MAX=h - 1,
        IN_W_MAX=w - 1,
        RUNNING_VAR_CORRECTION_C=RUNNING_VAR_CORRECTION,
        SCALE_H_C=SCALE_H,
        SCALE_W_C=SCALE_W,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=FINAL_C_BLOCK,
        num_warps=FINAL_WARPS,
        num_stages=3,
    )
    _cat_upsample_kernel[(triton.cdiv(cat.numel(), OUT_BLOCK),)](
        x,
        skip,
        weight,
        bias,
        mean,
        invstd,
        cat,
        C=c,
        IN_H=h,
        IN_W=w,
        OUT_H=out_h,
        OUT_W=out_w,
        TOTAL=cat.numel(),
        SCALE_H_C=SCALE_H,
        SCALE_W_C=SCALE_W,
        BLOCK=OUT_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=3,
    )
    return mean, invstd, y0, y1, x0, x1, x_weight, y_weight, cat, running_mean, running_var
