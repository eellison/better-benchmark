"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Google FNet f32 embedding assembly and LayerNorm inference scope in one Triton row kernel, including token-id embedding, expanded token-type embedding, position embedding, the two fp32 embedding adds, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 libdevice.rsqrt, affine epilogue, and the final contiguous `[16384,768]` view output, whereas Inductor lowers the decomposed embedding/expand/add/var_mean/rsqrt/affine/view graph through generic embedding, normalization, and pointwise schedules; Inductor cannot do this today because normalization-template canonicalization does not recognize this broadcast token-type plus positional embedding producer pattern as one semantic embedding-LayerNorm operation; the fix is NEW_PATTERN: add a guarded FNet embedding-LayerNorm template that folds the gathered embedding producers into a shape-specialized row-normalization kernel while preserving population variance, generated rsqrt semantics, and direct flat-output storage."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _fnet_embedding_layernorm_kernel(
    word_table_ptr,
    token_ids_ptr,
    token_type_ids_ptr,
    token_type_table_ptr,
    position_table_ptr,
    position_ids_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_ids = rows[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    offsets = row_ids * HIDDEN + cols
    seq = row_ids % SEQ_LEN

    token_id = tl.load(token_ids_ptr + row_ids, mask=row_mask, other=0)
    token_type_id = tl.load(token_type_ids_ptr + seq, mask=row_mask, other=0)
    position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)

    word = tl.load(
        word_table_ptr + token_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    token_type = tl.load(
        token_type_table_ptr + token_type_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    x = _f32_add(_f32_add(word, token_type), position)
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_masked = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
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
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(out_ptr + offsets, affine, mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 9a73f5a0: (T([32000,768], f32), T([32,512], i64), T([1,512], i64), T([4,768], f32), T([512,768], f32), T([1,512], i64), T([768], f32), T([768], f32), S([32,512]), S([16384,768]))
@oracle_impl(hardware="B200", point="9a73f5a0", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        _expand_shape,
        flat_shape,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq_len
    flat_shape = _as_shape(flat_shape)

    out = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=word_table.device,
        dtype=torch.float32,
    )

    _fnet_embedding_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        word_table,
        token_ids,
        token_type_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out
