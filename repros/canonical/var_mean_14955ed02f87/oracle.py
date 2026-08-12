"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 BERT embedding-plus-LayerNorm inference scope in one Triton row kernel, including the word embedding gather, token-type gather through the captured gather/expand indices, position embedding gather, the two observable bf16 embedding-add roundings, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 libdevice rsqrt, fp32 affine epilogue, final bf16 `[32,512,768]` output, and the three aliasing `[16384,768]` view returns, whereas Inductor lowers the embedding producers, bf16 add materialization, row normalization, affine cast, and repeated view returns through generic scheduler regions; Inductor cannot do this today because its fixed-hidden LayerNorm scheduler does not sink indexed embedding gathers and visible bf16 add boundaries into one full-scope row-normalization plan; the fix is SCHEDULER_FUSION: extend the embedding-LayerNorm template to fuse gather/add producers, preserve bf16 roundings, and emit the final buffer plus view aliases from one guarded schedule."""

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
def _embedding_layernorm_kernel(
    word_table_ptr,
    word_ids_ptr,
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
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (rows < ROWS) & (cols < HIDDEN)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    seq = rows - (rows // SEQ_LEN) * SEQ_LEN
    offsets = rows * HIDDEN + cols

    word_id = tl.load(word_ids_ptr + rows, mask=row_mask, other=0)
    position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)
    token_type_id = tl.load(token_type_source_ptr + position_id, mask=row_mask, other=0)

    word = tl.load(
        word_table_ptr + word_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    token_type = tl.load(
        token_type_table_ptr + token_type_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    position = tl.load(
        position_table_ptr + position_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)

    add0 = _f32_add(word.to(tl.float32), token_type.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    x = _f32_add(add0.to(tl.float32), position.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    ).to(tl.float32)

    x_masked = tl.where(mask, x, 0.0)
    mean = _f32_mul(tl.sum(x_masked, axis=1), 1.0 / HIDDEN)
    centered = _f32_sub(x, mean[:, None])
    var = _f32_mul(
        tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1),
        1.0 / HIDDEN,
    )
    invstd = libdevice.rsqrt(_f32_add(var, EPSILON))
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


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# (T([30522,768], bf16), T([32,512], i64), T([1,512], i64), T([1,512], i64), T([2,768], bf16), T([512,768], bf16), T([768], bf16), T([768], bf16), ...)
@oracle_impl(hardware="B200", point="a6271911", BLOCK_M=1, BLOCK_H=1024, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    (
        word_table,
        word_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        _token_type_shape,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs

    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len
    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        word_table,
        word_ids,
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
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        out,
        out.view(_as_shape(view_shape0)),
        out.view(_as_shape(view_shape1)),
        out.view(_as_shape(view_shape2)),
    )
