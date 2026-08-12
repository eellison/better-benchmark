"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ALBERT embedding composition and LayerNorm training scope in one fixed-hidden Triton row kernel, including the token-type `gather` output, broadcast expansion of the gathered token types across the batch, token/type/position embedding gathers, fp32 embedding adds, population `var_mean(..., dim=2, correction=0, keepdim=True)` over hidden size 128, eps=1e-12 rsqrt, returned normalized f32 tensor, affine epilogue with final bf16 cast/view, and `rsqrt / 128` side output, whereas Inductor lowers the decomposed gather/expand/embedding/add/normalization graph through generic indexing and normalization-template fragments; Inductor cannot do this today because norm-template canonicalization does not recognize ALBERT embedding assembly with a gathered token-type producer and sibling normalized/invstd outputs as one semantic fixed-K embedding-LayerNorm pattern; the fix is NEW_PATTERN: add an embedding-LayerNorm lowering that folds token, gathered type, and position indexed loads into the row-normalization kernel and emits the gathered token-type side output, normalized tensor, affine bf16 view, and inverse-std side output together."""

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
def _albert_embedding_layernorm_kernel(
    token_type_source_ptr,
    position_ids_ptr,
    token_table_ptr,
    token_ids_ptr,
    type_table_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    gather_out_ptr,
    normalized_ptr,
    affine_bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row_vec = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = row_vec[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    seq = rows - (rows // SEQ_LEN) * SEQ_LEN
    offsets = rows * HIDDEN + cols

    position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)
    token_type_id = tl.load(token_type_source_ptr + position_id, mask=row_mask, other=0)
    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)

    tl.store(
        gather_out_ptr + seq,
        token_type_id,
        mask=row_mask & (rows < SEQ_LEN),
    )

    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    token_type = tl.load(
        type_table_ptr + token_type_id * HIDDEN + cols,
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

    x = _f32_add(_f32_add(token, token_type), position)
    x_masked = tl.where(mask, x, 0.0)
    mean = _f32_mul(tl.sum(x_masked, axis=1), 1.0 / HIDDEN)
    centered_for_var = _f32_sub(x, mean[:, None])
    centered_masked = tl.where(mask, centered_for_var, 0.0)
    variance = _f32_mul(
        tl.sum(_f32_mul(centered_masked, centered_masked), axis=1),
        1.0 / HIDDEN,
    )
    centered = _f32_sub(x, mean[:, None])
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

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(affine_bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(div_ptr + row_vec, invstd / HIDDEN, mask=row_vec < ROWS)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="4c461b0d",
    BLOCK_M=4,
    BLOCK_H=128,
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
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        expand_shape,
        flat_shape,
    ) = inputs

    batch = int(expand_shape[0])
    seq_len = int(expand_shape[1])
    hidden = int(arg2_1.shape[1])
    rows = batch * seq_len
    flat_shape = _as_shape(flat_shape)

    gather = torch.empty_strided(
        (1, seq_len),
        (seq_len, 1),
        device=arg0_1.device,
        dtype=torch.int64,
    )
    normalized = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=arg2_1.device,
        dtype=torch.float32,
    )
    affine_bf16 = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        (batch, seq_len, 1),
        (seq_len, 1, 1),
        device=arg2_1.device,
        dtype=torch.float32,
    )

    _albert_embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        gather,
        normalized,
        affine_bf16,
        div,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return gather, normalized, affine_bf16, div
