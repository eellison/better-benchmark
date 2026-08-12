"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Google FNet inference residual LayerNorm scope in one Triton row kernel, including the `[16384,768] -> [32,512,768]` metadata view, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 `rsqrt`, affine scale/bias, and final contiguous `[16384,768]` view output, whereas Inductor lowers the single-use view/add producer and LayerNorm epilogue through a generic normalization schedule; Inductor cannot do this today because its fixed-hidden norm-template scheduler does not consistently inline the view/add producer and direct final-view store for this residual LayerNorm family; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to keep the reshaped add producer resident across the row statistics pass and emit the affine epilogue directly to the requested flat output layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 768
EPS = 1.0e-12


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
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS_C
    col_mask = cols < HIDDEN_C
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN_C + cols[None, :]

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(flat, residual)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN_C
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN_C
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

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
    out = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])
    tl.store(out_ptr + offsets, out, mask=mask)


# e864a84c: (T([16384,768], f32), T([32,512,768], f32), T([768], f32), T([768], f32), ...)
@oracle_impl(
    hardware="B200",
    point="e864a84c",
    BLOCK_H=1024,
    ROW_BLOCK=1,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1 = inputs
    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    _residual_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        ROWS_C=ROWS,
        HIDDEN_C=HIDDEN,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
