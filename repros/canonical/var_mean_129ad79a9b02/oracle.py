"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete CycleGAN bf16 instance-normalization ReLU scope for the captured N=1 NCHW tensors, including bf16-to-fp32 statistics, correction=0 per-channel population var_mean over `[N,H,W]`, eps=1e-5 rsqrt, fp32 normalization, the required bf16 rounding before ReLU, and the final contiguous bf16 output, whereas Inductor lowers the decomposed cast/var_mean/sub/rsqrt/mul/cast/relu graph through its generic normalization and pointwise schedules; Inductor cannot do this today because its scheduler does not select a guarded static CycleGAN instance-norm epilogue that preserves the bf16 cast boundary while writing the full activation directly; the fix is SCHEDULER_FUSION: add a fixed-N instance-norm ReLU template that fuses per-channel statistics with the dtype-aware ReLU epilogue for these static spatial planes."""

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
def _partial_channel_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    chunk = tl.program_id(1)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    elems = chunk * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = (channels[:, None] < C) & (elems[None, :] < HW)
    offsets = channels[:, None] * HW + elems[None, :]

    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    vals = tl.where(elems[None, :] < HW, vals, 0.0)
    sums = tl.sum(vals, axis=1)
    sumsqs = tl.sum(_f32_mul(vals, vals), axis=1)

    out_offsets = chunk * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sumsq_ptr + out_offsets, sumsqs, mask=channels < C)


@triton.jit
def _single_channel_norm_bf16_relu_kernel(
    x_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    channel = tl.program_id(0)
    elems = tl.arange(0, BLOCK_HW)
    mask = elems < HW
    offsets = channel * HW + elems

    x = tl.load(x_ptr + offsets, mask=(channel < C) & mask, other=0.0).to(tl.float32)
    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=0) / HW
    centered = _f32_sub(x, mean)
    centered_for_sum = tl.where(mask, centered, 0.0)
    var = tl.sum(_f32_mul(centered_for_sum, centered_for_sum), axis=0) / HW
    invstd = libdevice.rsqrt(_f32_add(tl.maximum(var, 0.0), 1.0e-5))
    y = _f32_mul(centered, invstd)
    rounded = y.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu = _relu_preserve_nan(rounded)
    tl.store(out_ptr + offsets, relu.to(tl.bfloat16), mask=(channel < C) & mask)


@triton.jit
def _final_channel_stats_kernel(
    partial_sum_ptr,
    partial_sumsq_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]

    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sumsqs = tl.load(partial_sumsq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(sums, axis=1)
    total_sq = tl.sum(sumsqs, axis=1)
    mean = total / HW
    var = _f32_sub(total_sq / HW, _f32_mul(mean, mean))
    var = tl.maximum(var, 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    channel_mask = channels < C
    tl.store(mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)


@triton.jit
def _norm_bf16_relu_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < TOTAL
    channels = offsets // HW

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    y = _f32_mul(_f32_sub(x, mean), invstd)
    rounded = y.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu = _relu_preserve_nan(rounded)
    tl.store(out_ptr + offsets, relu.to(tl.bfloat16), mask=mask)


def _launch(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    FINAL_C_BLOCK: int,
    OUT_BLOCK: int,
    STAT_WARPS: int,
    FINAL_WARPS: int,
    OUT_WARPS: int,
    SINGLE_TILE: bool,
    BLOCK_HW: int,
    SINGLE_WARPS: int,
    num_stages: int,
):
    (x,) = inputs
    _batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    total = channels * hw
    out = torch.empty_strided(
        (1, channels, height, width),
        (channels * hw, hw, width, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    if SINGLE_TILE:
        _single_channel_norm_bf16_relu_kernel[(channels,)](
            x,
            out,
            C=channels,
            HW=hw,
            BLOCK_HW=BLOCK_HW,
            num_warps=SINGLE_WARPS,
            num_stages=num_stages,
        )
        return out

    num_chunks = triton.cdiv(hw, BLOCK_E)
    block_chunks = triton.next_power_of_2(num_chunks)
    partial_sum = torch.empty((num_chunks, channels), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty_like(partial_sum)
    mean = torch.empty((channels,), device=x.device, dtype=torch.float32)
    invstd = torch.empty((channels,), device=x.device, dtype=torch.float32)

    _partial_channel_stats_kernel[(triton.cdiv(channels, C_BLOCK), num_chunks)](
        x,
        partial_sum,
        partial_sumsq,
        C=channels,
        HW=hw,
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=STAT_WARPS,
        num_stages=num_stages,
    )
    _final_channel_stats_kernel[(triton.cdiv(channels, FINAL_C_BLOCK),)](
        partial_sum,
        partial_sumsq,
        mean,
        invstd,
        C=channels,
        HW=hw,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=FINAL_C_BLOCK,
        num_warps=FINAL_WARPS,
        num_stages=num_stages,
    )
    _norm_bf16_relu_kernel[(triton.cdiv(total, OUT_BLOCK),)](
        x,
        mean,
        invstd,
        out,
        TOTAL=total,
        HW=hw,
        BLOCK_E=OUT_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=num_stages,
    )
    return out


# (T([1,128,128,128], bf16))
@oracle_impl(hardware="B200", point="a9dd97a3", BLOCK_E=1024, C_BLOCK=1, FINAL_C_BLOCK=8, OUT_BLOCK=1024, STAT_WARPS=4, FINAL_WARPS=1, OUT_WARPS=4, SINGLE_TILE=True, BLOCK_HW=16384, SINGLE_WARPS=8, num_stages=3)
# (T([1,64,256,256], bf16))
@oracle_impl(hardware="B200", point="b9dd6ddf", BLOCK_E=1024, C_BLOCK=1, FINAL_C_BLOCK=8, OUT_BLOCK=1024, STAT_WARPS=4, FINAL_WARPS=1, OUT_WARPS=4, SINGLE_TILE=False, BLOCK_HW=0, SINGLE_WARPS=4, num_stages=3)
def oracle_forward(inputs, **kwargs):
    return _launch(inputs, **kwargs)
