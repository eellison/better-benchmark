"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LayoutLM bf16 embedding-plus-LayerNorm inference scope in one Triton row kernel, including word embedding gather, learned-position gather, the six constant-zero spatial/token-type embedding rows, eager-visible bf16 add boundaries, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, fp32 scale/bias affine epilogue, final bf16 `[32,512,768]` output, and the three aliasing `[16384,768]` view returns, whereas Inductor lowers the constant-id producers, embedding gathers, row normalization, affine cast, and repeated view aliases through generic embedding and normalization schedules; Inductor cannot do this today because its fixed-hidden normalization scheduler does not recognize LayoutLM's multi-table constant-id embedding assembly as a row-local LayerNorm producer while preserving bf16 rounding boundaries and alias-only returns; the fix is SCHEDULER_FUSION: extend the embedding-LayerNorm template to fold constant-id embedding rows and indexed word/position gathers into the row-normalization kernel and emit the base bf16 tensor plus shared view aliases directly."""

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
def _bf16_add_boundary(a, b):
    return _f32_add(a, b).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    ).to(tl.float32)


@triton.jit
def _layoutlm_embedding_layernorm_kernel(
    word_table_ptr,
    word_ids_ptr,
    position_table_ptr,
    position_ids_ptr,
    x_position_table_ptr,
    y_position_table_ptr,
    h_position_table_ptr,
    w_position_table_ptr,
    token_type_table_ptr,
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

    word_id = tl.load(word_ids_ptr + rows, mask=row_mask, other=0)
    position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)

    word = tl.load(
        word_table_ptr + word_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x_position = tl.load(
        x_position_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    y_position = tl.load(
        y_position_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    h_position = tl.load(
        h_position_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    w_position = tl.load(
        w_position_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    token_type = tl.load(
        token_type_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    if USE_INDUCTOR_NUMERICS:
        x = _f32_add(word, position)
        x = _f32_add(x, x_position)
        x = _f32_add(x, y_position)
        x = _f32_add(x, x_position)
        x = _f32_add(x, y_position)
        x = _f32_add(x, h_position)
        x = _f32_add(x, w_position)
        x = _f32_add(x, token_type)
    else:
        x = _bf16_add_boundary(word, position)
        x = _bf16_add_boundary(x, x_position)
        x = _bf16_add_boundary(x, y_position)
        x = _bf16_add_boundary(x, x_position)
        x = _bf16_add_boundary(x, y_position)
        x = _bf16_add_boundary(x, h_position)
        x = _bf16_add_boundary(x, w_position)
        x = _bf16_add_boundary(x, token_type)

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


# 771dc86a: LayoutLMForMaskedLM infer embedding + LayerNorm, hidden=768, rows=16384.
@oracle_impl(
    hardware="B200",
    point="771dc86a",
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
        position_table,
        position_ids,
        x_position_table,
        y_position_table,
        h_position_table,
        w_position_table,
        token_type_table,
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

    _layoutlm_embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        word_table,
        word_ids,
        position_table,
        position_ids,
        x_position_table,
        y_position_table,
        h_position_table,
        w_position_table,
        token_type_table,
        weight,
        bias,
        out,
        ROWS=rows,
        SEQ_LEN=seq_len,
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
