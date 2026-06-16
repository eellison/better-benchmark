"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Swin residual bf16 add, fixed-width population var_mean over 1024 channels, eps=1e-5 rsqrt affine epilogue, required bf16 materialization, and final 7x7 spatial mean in two Triton kernels, whereas Inductor lowers the captured add/cast/var_mean/affine/cast/mean graph through generic normalization and downstream reduction schedules; Inductor cannot do this today because its scheduler does not keep the bf16 residual-add boundary and the consuming spatial mean inside one fixed-shape normalization plan; the fix is SCHEDULER_FUSION: extend the layernorm template to preserve strict dtype boundaries while sinking the static 49-element mean consumer into the same full-scope lowering."""

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
def _residual_layernorm_kernel(
    arg0_ptr,
    arg1_ptr,
    weight_ptr,
    bias_ptr,
    norm_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    offsets = rows * HIDDEN + cols

    residual = tl.load(
        arg0_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    skip = tl.load(
        arg1_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(skip, residual).to(tl.bfloat16).to(tl.float32)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(
        _f32_mul(centered_for_var, centered_for_var),
        axis=1,
    )[:, None] / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    y = _f32_add(_f32_mul(_f32_mul(centered, invstd), weight), bias).to(tl.bfloat16)
    tl.store(norm_ptr + offsets, y, mask=mask)


@triton.jit
def _spatial_mean_kernel(
    norm_ptr,
    out_ptr,
    BATCH: tl.constexpr,
    SPATIAL: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    spatial = tl.arange(0, BLOCK_S)[None, :]
    channel_mask = channels < HIDDEN
    spatial_mask = spatial < SPATIAL
    mask = channel_mask & spatial_mask
    offsets = (batch * SPATIAL + spatial) * HIDDEN + channels

    vals = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(tl.where(mask, vals, 0.0), axis=1)
    mean = (sums / SPATIAL).to(tl.bfloat16)
    out_offsets = batch * HIDDEN + channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(out_ptr + out_offsets, mean, mask=(channel_block * BLOCK_C + tl.arange(0, BLOCK_C)) < HIDDEN)


# 1a2bb10a: (T([6272,1024], bf16), T([128,49,1024], bf16), T([1024], bf16), T([1024], bf16), ...)
@oracle_impl(
    hardware="B200",
    point="1a2bb10a",
    BLOCK_N=1024,
    ROW_BLOCK=2,
    BLOCK_C=32,
    BLOCK_S=64,
    LN_WARPS=8,
    MEAN_WARPS=1,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    ROW_BLOCK: int,
    BLOCK_C: int,
    BLOCK_S: int,
    LN_WARPS: int,
    MEAN_WARPS: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1 = inputs
    batch = int(arg1_1.shape[0])
    spatial = int(arg1_1.shape[1])
    hidden = int(arg1_1.shape[2])
    rows = batch * spatial

    norm = torch.empty_strided(
        (batch, spatial, hidden),
        (spatial * hidden, hidden, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        (batch, hidden),
        (hidden, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        norm,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_N=BLOCK_N,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=LN_WARPS,
        num_stages=3,
    )
    _spatial_mean_kernel[(batch, triton.cdiv(hidden, BLOCK_C))](
        norm,
        out,
        BATCH=batch,
        SPATIAL=spatial,
        HIDDEN=hidden,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=MEAN_WARPS,
        num_stages=1,
    )
    return out
