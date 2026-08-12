"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistillGPT2 bf16 token-plus-generated-position embedding LayerNorm scope in one Triton row kernel, including token embedding gather, generated position embedding gather, required bf16 embedding-add materialization returned as `[32,512,768]`, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` over the rounded add, eps=1e-5 rsqrt affine epilogue, final bf16 `[16384,768]` view, and the adjacent-position bool side output as an all-false store, whereas Inductor lowers the embedding/add/var_mean/affine/view graph and the sibling cat/slice/sub/ne mask through generic embedding, normalization, and pointwise schedules; Inductor cannot do this today because its fixed-hidden normalization scheduler does not fuse generated-position embedding producers, observable pre-norm bf16 materialization, row statistics, affine cast/view, and algebraically constant sibling mask output into one full-scope template; the fix is NEW_PATTERN: add a GPT-style embedding-LayerNorm lowering that folds indexed token/position loads into the row kernel, preserves the bf16 add/output boundaries, and emits the constant adjacent-position mask from the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


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
def _embedding_layernorm_mask_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    mask_out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    token_id = tl.load(token_ids_ptr + row_ids, mask=row_mask, other=0)[:, None]
    position_id = (row_ids % SEQ_LEN)[:, None]
    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    add_bf16 = _f32_add(token, position).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)
    x = add_bf16.to(tl.float32)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))
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
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])
    tl.store(norm_out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(mask_out_ptr + row_ids, False, mask=row_mask)


# 7a2cdb11: (T([50257,768], bf16), T([32,512], i64), T([1024,768], bf16), T([768], bf16), T([768], bf16), ...)
@oracle_impl(hardware="B200", point="7a2cdb11", BLOCK_M=2, BLOCK_H=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int, num_warps: int, num_stages: int):
    token_table, token_ids, position_table, weight, bias, _shape0, _shape1 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    add_out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    mask_out = torch.empty_strided(
        (batch, seq_len),
        (seq_len, 1),
        device=token_table.device,
        dtype=torch.bool,
    )

    _embedding_layernorm_mask_kernel[(triton.cdiv(rows, BLOCK_M),)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        add_out,
        norm_out,
        mask_out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out, mask_out
