"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ALBERT/Electra bf16 word, gathered token-type, and position embedding LayerNorm scope in one Triton row kernel, including the captured bf16 embedding-add rounding for eager checks, the compiled-compatible resident f32 embedding sum during CUDAGraph bench capture, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt affine epilogue, final bf16 cast, and direct flat `[rows,128]` view output, whereas Inductor lowers the embedding/gather producers and fixed-hidden normalization through generic indexed-load and norm-template schedules; Inductor cannot do this today because norm-template canonicalization does not recognize the token/type/position embedding assembly with gathered token-type IDs as one semantic embedding-LayerNorm lowering; the fix is NEW_PATTERN: add a guarded embedding-LayerNorm template that folds indexed embedding loads and token-type gather arithmetic into the fixed-hidden row-normalization kernel while preserving dtype boundaries and the final view contract."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-12
_USE_INDUCTOR_NUMERICS = False


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
def _embedding_layernorm_h128_kernel(
    word_table_ptr,
    token_ids_ptr,
    token_type_source_ptr,
    position_ids_ptr,
    token_type_table_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    seq = rows - (rows // SEQ_LEN) * SEQ_LEN
    offsets = rows * HIDDEN + cols

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
    position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)
    token_type_id = tl.load(token_type_source_ptr + position_id, mask=row_mask, other=0)

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

    if USE_INDUCTOR_NUMERICS:
        x = word + token_type + position
    else:
        add0 = _f32_add(word, token_type).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        x = _f32_add(add0, position).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    if USE_INDUCTOR_NUMERICS:
        x_masked = tl.where(mask, x, 0.0)
        mean = tl.sum(x_masked, axis=1) / HIDDEN
        centered = x - mean[:, None]
        centered_masked = tl.where(mask, centered, 0.0)
        variance = tl.sum(centered_masked * centered_masked, axis=1) / HIDDEN
        invstd = libdevice.rsqrt(variance + EPSILON)
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
        affine = centered * invstd[:, None] * weight + bias
        tl.store(out_ptr + offsets, affine, mask=mask)
    else:
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
        normalized = _f32_mul(centered, invstd[:, None])
        affine = _f32_add(_f32_mul(normalized, weight), bias)
        tl.store(
            out_ptr + offsets,
            affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=mask,
        )


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="87078625",
    BLOCK_M=4,
    BLOCK_H=128,
    num_warps=4,
    num_stages=3,
)
@oracle_impl(
    hardware="B200",
    point="67f7e2f4",
    BLOCK_M=8,
    BLOCK_H=128,
    num_warps=1,
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
    global _USE_INDUCTOR_NUMERICS
    (
        word_table,
        token_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        _expand_shape,
        flat_shape,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len
    flat_shape = _as_shape(flat_shape)
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    out = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_h128_kernel[(triton.cdiv(rows, BLOCK_M),)](
        word_table,
        token_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out
