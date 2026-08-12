"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete CycleGAN bf16 instance-normalization ReLU and two-axis reflected unsafe-index scope for the captured `[1, 64, 256, 256]` tensor, including bf16-to-fp32 population var_mean over `[N,H,W]`, correction=0, eps=1e-5 rsqrt, fp32 normalization, the required bf16 rounding before ReLU, and a direct contiguous bf16 `[1, 64, 262, 262]` reflected output, whereas Inductor schedules the decomposed normalization, dtype-boundary ReLU, and reflection indexing as separate generic regions; Inductor cannot do this today because its scheduler does not fuse static reflected-index consumers into the dtype-aware instance-normalization epilogue; the fix is SCHEDULER_FUSION: add a fixed-shape reflected-layout instance-norm ReLU template that preserves the bf16 cast boundary while writing the final padded layout directly."""

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
def _reflect_norm_bf16_relu_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    HW: tl.constexpr,
    OUT_HW: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < TOTAL
    channels = offsets // OUT_HW
    spatial = offsets - channels * OUT_HW
    oh = spatial // (HEIGHT + PAD + PAD)
    ow = spatial - oh * (WIDTH + PAD + PAD)

    h_shifted = oh - PAD
    h_abs = tl.where(h_shifted < 0, -h_shifted, h_shifted)
    h_folded = (HEIGHT - 1) - h_abs
    h_folded_abs = tl.where(h_folded < 0, -h_folded, h_folded)
    ih = (HEIGHT - 1) - h_folded_abs

    w_shifted = ow - PAD
    w_abs = tl.where(w_shifted < 0, -w_shifted, w_shifted)
    w_folded = (WIDTH - 1) - w_abs
    w_folded_abs = tl.where(w_folded < 0, -w_folded, w_folded)
    iw = (WIDTH - 1) - w_folded_abs

    input_offsets = channels * HW + ih * WIDTH + iw
    x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    y = _f32_mul(_f32_sub(x, mean), invstd)
    rounded = y.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    relu = _relu_preserve_nan(rounded)
    tl.store(out_ptr + offsets, relu.to(tl.bfloat16), mask=mask)


# (T([1,64,256,256], bf16))
@oracle_impl(
    hardware="B200",
    point="b9dd6ddf",
    BLOCK_E=1024,
    C_BLOCK=1,
    FINAL_C_BLOCK=8,
    OUT_BLOCK=1024,
    STAT_WARPS=4,
    FINAL_WARPS=1,
    OUT_WARPS=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    FINAL_C_BLOCK: int,
    OUT_BLOCK: int,
    STAT_WARPS: int,
    FINAL_WARPS: int,
    OUT_WARPS: int,
    num_stages: int,
):
    (x,) = inputs
    channels = 64
    height = 256
    width = 256
    pad = 3
    hw = height * width
    padded = height + pad + pad
    out_hw = padded * padded
    total = channels * out_hw
    out = torch.empty_strided(
        (1, channels, padded, padded),
        (channels * out_hw, out_hw, padded, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

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
    _reflect_norm_bf16_relu_kernel[(triton.cdiv(total, OUT_BLOCK),)](
        x,
        mean,
        invstd,
        out,
        TOTAL=total,
        HW=hw,
        OUT_HW=out_hw,
        HEIGHT=height,
        WIDTH=width,
        PAD=pad,
        BLOCK_E=OUT_BLOCK,
        num_warps=OUT_WARPS,
        num_stages=num_stages,
    )
    return out
