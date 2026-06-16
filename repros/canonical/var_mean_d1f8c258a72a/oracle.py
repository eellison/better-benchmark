"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete RoBERTa bf16 word/token-type/position embedding LayerNorm inference scope in one Triton row kernel, including the word embedding gather, the expanded-and-gathered token-type ids from the captured int32 product index, the position embedding gather from that same product index, the eager-visible bf16 embedding-add boundaries, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, bf16 scale/bias affine epilogue, final bf16 `[32,512,768]` output, and the three aliasing `[16384,768]` view returns, whereas Inductor lowers the embedding gathers, gather/expand index arithmetic, row reduction, affine cast, and repeated view aliases through generic embedding and normalization schedules; Inductor cannot do this today because the fixed-hidden normalization scheduler does not recognize this RoBERTa token/type/position embedding assembly as a semantic row-local LayerNorm producer while preserving bf16 rounding boundaries and alias-only view returns; the fix is NEW_PATTERN: add a guarded RoBERTa embedding-LayerNorm lowering that folds indexed embedding loads and token-type gather arithmetic into the row-normalization kernel and emits the base bf16 tensor plus shared view aliases directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _roberta_embedding_layernorm_kernel(
    word_table_ptr,
    word_ids_ptr,
    token_type_source_ptr,
    position_index_source_ptr,
    position_mask_ptr,
    token_type_table_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
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
    offsets = rows * HIDDEN + cols

    word_id = tl.load(word_ids_ptr + rows, mask=row_mask, other=0)
    raw_index = tl.load(position_index_source_ptr + rows, mask=row_mask, other=0)
    index_i32 = raw_index.to(tl.int32)
    index_mask = tl.load(position_mask_ptr + rows, mask=row_mask, other=0)
    position_id = (index_i32 * index_mask).to(tl.int64)
    token_type_id = tl.load(token_type_source_ptr + position_id, mask=row_mask, other=0)

    word = tl.load(
        word_table_ptr + word_id * HIDDEN + cols,
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
        x = _f32_add(_f32_add(word, token_type), position)
    else:
        add0 = _f32_add(word, token_type).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        ).to(tl.float32)
        x = _f32_add(add0, position).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        ).to(tl.float32)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = _f32_mul(tl.sum(x_for_reduce, axis=1), 1.0 / HIDDEN)
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = _f32_mul(
        tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1),
        1.0 / HIDDEN,
    )
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
    tl.store(
        out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 975c5fbc: RoBERTaForCausalLM infer embedding + LayerNorm, hidden=768, rows=16384.
@oracle_impl(
    hardware="B200",
    point="975c5fbc",
    BLOCK_M=2,
    BLOCK_H=1024,
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
    global _USE_INDUCTOR_NUMERICS
    (
        word_table,
        word_ids,
        token_type_source,
        position_index_source,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        _shape0,
        _shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs
    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )

    _roberta_embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        word_table,
        word_ids,
        token_type_source,
        position_index_source,
        position_mask,
        token_type_table,
        position_table,
        weight,
        bias,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPSILON=1.0e-12,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        out,
        out.view(_shape_tuple(shape2)),
        out.view(_shape_tuple(shape3)),
        out.view(_shape_tuple(shape4)),
    )
