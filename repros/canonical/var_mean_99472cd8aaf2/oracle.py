"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 SigLIP inference hidden-size-768 LayerNorm scope in one shape-specialized Triton row kernel, including the metadata-only returned `[128,1,768]` input view, bf16-to-fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, bf16 affine scale/bias promoted through the fp32 epilogue, final bf16 cast, and returned contiguous `[128,768]` view, whereas Inductor already lowers this small fixed-hidden normalization through its generic norm template; Inductor cannot materially improve this local repro through producer fusion, split-K, recompute, or alias-only elimination because the remaining work is the required activation/affine traffic, row reduction, and output write; the fix is BANDWIDTH_BOUND: record this as an at-floor normalization case unless broader LayerNorm codegen or launch-overhead work moves both paths."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-6


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
def _layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids[:, None] < ROWS
    col_mask = cols[None, :] < HIDDEN
    mask = row_mask & col_mask
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    values = tl.where(mask, x, 0.0)
    m2 = tl.zeros((ROW_BLOCK, BLOCK_H), tl.float32)
    weights_for_reduce = tl.where(mask, 1.0, 0.0)
    mean_1d, m2_1d, _ = triton_helpers.welford(values, m2, weights_for_reduce, 1)
    variance_1d = m2_1d / HIDDEN
    invstd_1d = libdevice.rsqrt(_f32_add(variance_1d, EPS_C))

    centered = _f32_sub(x, mean_1d[:, None])
    normalized = _f32_mul(centered, invstd_1d[:, None])

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

    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 53185b1e: (T([128,768], bf16), T([768], bf16), T([768], bf16), S([128,1,768]), S([128,768]))
@oracle_impl(hardware="B200", point="53185b1e", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    rows = int(out_shape[0])
    hidden = int(out_shape[1])

    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return arg0_1.view(view_shape), out
