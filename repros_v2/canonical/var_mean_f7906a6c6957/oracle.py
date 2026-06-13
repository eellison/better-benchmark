"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GenAI static LayerNorm-forward scope in one multi-row Triton normalization kernel, including the bf16 input promotion to fp32, population `var_mean(..., dim=1, correction=0, keepdim=True)`, eps=1e-6 `rsqrt`, fp32 weight multiply, final bf16 rounding, and contiguous `[1152000,512]` output layout. Inductor lowers the decomposed cast/var_mean/sub/rsqrt/mul/mul/cast graph through its generic row-reduction template, which does not expose the shape-specialized hidden-size-512 multi-row schedule and affine bf16 epilogue as one guarded full-scope plan. The fix is SCHEDULER_FUSION: add a fixed-hidden LayerNorm-forward schedule that chooses the multi-row tile and sinks the weight multiply plus bf16 conversion into the normalization epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-6


@triton.jit
def _layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / HIDDEN
    centered = x - mean[:, None]
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(variance + EPS_C)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    out = centered * invstd[:, None] * weight[None, :]
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


# e5ae55b5: (T([1152000,512], bf16), T([512], f32))
@oracle_impl(
    hardware="B200",
    point="e5ae55b5",
    BLOCK_M=8,
    BLOCK_H=512,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    x, weight = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _layernorm_forward_kernel[(triton.cdiv(rows, BLOCK_M),)](
        x,
        weight,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
