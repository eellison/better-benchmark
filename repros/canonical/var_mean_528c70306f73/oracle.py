"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete mixed bf16/fp32 residual LayerNorm scope in one shape-specialized Triton row kernel, including the flat bf16 input view, fp32 residual add, correction=0 Welford var_mean over the hidden dimension, `rsqrt(var + 1e-6)`, returned normalized fp32 tensor, affine scale/bias epilogue, final bf16 view, and sibling `rsqrt / 768` output, whereas Inductor lowers the decomposed add/var_mean/normalized-side-output/affine/cast/view graph through its generic normalization schedule; Inductor cannot do this today because the normalization scheduler does not keep the mixed-dtype producer tile resident while emitting all live normalized, rounded-affine, and saved-scale epilogues in one full-scope row plan; the fix is SCHEDULER_FUSION: extend the LayerNorm template to fuse mixed bf16/fp32 residual producers with multi-output epilogues while preserving correction=0 Welford numerics, the captured saved-scale divisor literal, and bf16 cast boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
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
def _mixed_residual_layernorm_kernel(
    flat_bf16_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    final_bf16_ptr,
    invstd_div_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SIDE_DIVISOR: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids[:, None] < ROWS
    col_mask = cols[None, :] < HIDDEN
    mask = row_mask & col_mask
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    flat = tl.load(
        flat_bf16_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(residual, flat)

    values = tl.where(mask, x, 0.0)
    m2 = tl.zeros((ROW_BLOCK, BLOCK_H), tl.float32)
    weights = tl.where(mask, 1.0, 0.0)
    mean_1d, m2_1d, _ = triton_helpers.welford(values, m2, weights, 1)
    mean = mean_1d[:, None]
    variance_1d = m2_1d / HIDDEN
    invstd_1d = libdevice.rsqrt(_f32_add(variance_1d, 1.0e-6))
    invstd = invstd_1d[:, None]

    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(final_bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(
        invstd_div_ptr + row_ids,
        invstd_1d / SIDE_DIVISOR,
        mask=row_ids < ROWS,
    )


@oracle_impl(hardware="B200", point="7b097b88", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="5d43e450", BLOCK_H=2048, ROW_BLOCK=1, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    norm_shape = tuple(int(dim) for dim in shape0)
    final_shape = tuple(int(dim) for dim in shape1)
    batch = int(norm_shape[0])
    seq_len = int(norm_shape[1])
    hidden = int(norm_shape[2])
    rows = batch * seq_len

    normalized = torch.empty_strided(
        norm_shape,
        (seq_len * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        final_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (batch, seq_len, 1),
        (seq_len, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _mixed_residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        normalized,
        final_bf16,
        invstd_div,
        ROWS=rows,
        HIDDEN=hidden,
        SIDE_DIVISOR=768,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return normalized, final_bf16, invstd_div
