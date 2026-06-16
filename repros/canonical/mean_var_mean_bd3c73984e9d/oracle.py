"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BEiT bf16-input residual-MLP view, fp32 channel broadcast multiply, fp32 residual add, class-token drop, 196-token pooled mean, fp32 hidden-size-768 population LayerNorm, returned normalized tensor, bf16 affine output, and returned `rsqrt / 768` side output with a specialized pooled-token reduction feeding a row-wise normalization epilogue, whereas Inductor schedules the pooled-token mean producer and the norm-template consumer as separate generic regions around a compact intermediate while also materializing the visible side outputs; Inductor cannot do this today because its normalization-template scheduler does not sink a preceding fixed-token reduction producer into the LayerNorm row-statistics plan when the producer has fp32 broadcast arithmetic and multiple returned epilogue values; the fix is SCHEDULER_FUSION: teach the scheduler to fuse fixed-token pooling reductions into static hidden-size LayerNorm kernels while preserving fp32 intermediates, bf16 output casting, strides, and side outputs."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _pooled_sum_kernel(
    mlp_ptr,
    gamma_ptr,
    residual_ptr,
    pooled_ptr,
    TOTAL: tl.constexpr,
    TOKENS: tl.constexpr,
    LIVE_TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xidx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    ridx = tl.arange(0, RBLOCK)[None, :]
    xmask = xidx < TOTAL
    channel = xidx % HIDDEN
    batch = xidx // HIDDEN

    gamma = tl.load(
        gamma_ptr + channel,
        mask=xmask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    accum = tl.zeros((XBLOCK, RBLOCK), dtype=tl.float32)
    for roffset in tl.range(0, LIVE_TOKENS, RBLOCK):
        token = 1 + roffset + ridx
        tmask = token < TOKENS
        offsets = (batch * TOKENS + token) * HIDDEN + channel
        mlp = tl.load(
            mlp_ptr + offsets,
            mask=xmask & tmask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=xmask & tmask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        value = residual + gamma * mlp
        accum += tl.where(tmask, value, 0.0)

    pooled_sum = tl.sum(accum, axis=1)[:, None]
    tl.store(pooled_ptr + xidx, pooled_sum, mask=xmask)


@triton.jit
def _layernorm_outputs_kernel(
    pooled_ptr,
    weight_ptr,
    bias_ptr,
    normed_ptr,
    out_ptr,
    side_ptr,
    HIDDEN: tl.constexpr,
    LIVE_TOKENS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    pooled_sum = tl.load(
        pooled_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    pooled = pooled_sum / LIVE_TOKENS
    mean = tl.sum(tl.where(mask, pooled, 0.0), axis=0) / HIDDEN
    centered = tl.where(mask, pooled - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / HIDDEN
    invstd = libdevice.rsqrt(variance + 1.0e-6)
    normed = centered * invstd

    weight = tl.load(
        weight_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = normed * weight + bias

    tl.store(normed_ptr + offsets, normed, mask=mask)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(side_ptr + row, invstd / HIDDEN)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# f4c82f7a: (T([25216,768], bf16), T([768], f32), T([128,197,768], f32), T([768], f32), T([768], f32), S([128,197,768]))
@oracle_impl(
    hardware="B200",
    point="f4c82f7a",
    XBLOCK=64,
    RBLOCK=8,
    BLOCK_H=1024,
    pool_warps=4,
    norm_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    XBLOCK: int,
    RBLOCK: int,
    BLOCK_H: int,
    pool_warps: int,
    norm_warps: int,
    num_stages: int,
):
    mlp, gamma, residual, weight, bias, view_shape = inputs
    batch, tokens, hidden = _shape_tuple(view_shape)
    live_tokens = tokens - 1
    total = batch * hidden

    out_shape = (batch, hidden)
    out_stride = _contiguous_stride(out_shape)
    side_shape = (batch, 1)

    pooled = torch.empty_strided(
        out_shape,
        out_stride,
        device=mlp.device,
        dtype=torch.float32,
    )
    normed = torch.empty_strided(
        out_shape,
        out_stride,
        device=mlp.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=mlp.device,
        dtype=torch.bfloat16,
    )
    side = torch.empty_strided(
        side_shape,
        _contiguous_stride(side_shape),
        device=mlp.device,
        dtype=torch.float32,
    )

    _pooled_sum_kernel[(triton.cdiv(total, XBLOCK),)](
        mlp,
        gamma,
        residual,
        pooled,
        TOTAL=total,
        TOKENS=tokens,
        LIVE_TOKENS=live_tokens,
        HIDDEN=hidden,
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
        num_warps=pool_warps,
        num_stages=num_stages,
    )
    _layernorm_outputs_kernel[(batch,)](
        pooled,
        weight,
        bias,
        normed,
        out,
        side,
        HIDDEN=hidden,
        LIVE_TOKENS=live_tokens,
        BLOCK_H=BLOCK_H,
        num_warps=norm_warps,
        num_stages=num_stages,
    )
    return normed, out, side
