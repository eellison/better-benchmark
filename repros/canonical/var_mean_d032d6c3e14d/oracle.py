"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DeiT residual-add LayerNorm scope for the two observable token selects, including the `[25344,768] -> [128,198,768]` metadata view, Inductor's resident fp32 residual-add path for the unreturned producer, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 libdevice rsqrt, bf16 scale/bias affine epilogue, final bf16 cast, and the returned token-0/token-1 views sharing a full-stride `[128,198,768]` backing tensor, whereas Inductor lowers the row-independent normalization over all 25,344 token rows before applying the two `select` consumers; Inductor cannot do this today because its scheduler does not push fixed token selects backward through view, residual add, var_mean, rsqrt, affine, and cast to prove the other 196 token rows per batch are dead; the fix is ALGEBRAIC_ELIMINATION: commute constant token selects through row-local normalization graphs and narrow the scheduled row domain before codegen."""

import torch
import triton
import triton.language as tl
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
def _selected_token_pair_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    TOKENS: tl.constexpr,
    SELECTED_TOKENS: tl.constexpr,
    SELECTED_ROWS: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    selected_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = selected_rows[:, None] < SELECTED_ROWS
    col_mask = cols[None, :] < HIDDEN
    mask = row_mask & col_mask

    batch = selected_rows // SELECTED_TOKENS
    token = selected_rows - batch * SELECTED_TOKENS
    source_rows = batch * TOKENS + token
    offsets = source_rows[:, None] * HIDDEN + cols[None, :]

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    added = _f32_add(flat.to(tl.float32), residual.to(tl.float32))

    values = tl.where(mask, added, 0.0)
    mean = tl.sum(values, axis=1)[:, None] / HIDDEN
    mean_sq = tl.sum(_f32_mul(values, values), axis=1)[:, None] / HIDDEN
    variance = _f32_sub(mean_sq, _f32_mul(mean, mean))
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    centered = _f32_sub(added, mean)
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
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 155170ab: (T([25344,768], bf16), T([128,198,768], bf16), T([768], bf16), T([768], bf16), S([128,198,768]))
@oracle_impl(hardware="B200", point="155170ab", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    base_shape = _as_shape(shape0)
    batch = int(base_shape[0])
    tokens = int(base_shape[1])
    hidden = int(base_shape[2])
    selected_tokens = 2
    selected_rows = batch * selected_tokens

    out_base = torch.empty_strided(
        base_shape,
        (tokens * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _selected_token_pair_layernorm_kernel[(triton.cdiv(selected_rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out_base,
        HIDDEN=hidden,
        TOKENS=tokens,
        SELECTED_TOKENS=selected_tokens,
        SELECTED_ROWS=selected_rows,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_base.select(1, 0), out_base.select(1, 1)
