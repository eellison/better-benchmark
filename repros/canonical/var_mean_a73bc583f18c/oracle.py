"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DeBERTa word/position embedding plus LayerNorm inference scope in one Triton row kernel, including the word and position embedding gathers, the captured bf16 embedding-add rounding before fp32 promotion, population `var_mean(..., dim=2, correction=0, keepdim=True)` with eps=1e-7 `libdevice.rsqrt`, fp32 affine with bf16 scale/bias promotion, final bf16 cast, and the three aliasing `[4096,1536]` view returns from the same `[8,512,1536]` output storage, whereas Inductor lowers the indexed embedding producers, bf16 add materialization, row reduction, affine cast, and repeated view aliases through generic scheduler regions; Inductor cannot fuse this full returned-output envelope today because the fixed-hidden normalization template does not sink token/position embedding gathers through the bf16 add boundary while preserving the final storage aliases; the fix is SCHEDULER_FUSION: extend the embedding-LayerNorm template to fuse gathered producers into the row-normalization plan and emit the base bf16 tensor plus alias-only views directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-7


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
def _deberta_embedding_layernorm_kernel(
    word_table_ptr,
    word_ids_ptr,
    position_table_ptr,
    position_ids_ptr,
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
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    offsets = rows * HIDDEN + cols
    seq = rows % SEQ_LEN

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
        eviction_policy="evict_last",
    ).to(tl.float32)

    x = _f32_add(word, position).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
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


# 20cc819f: (T([128100,1536], bf16), T([8,512], i64), T([512,1536], bf16), T([1,512], i64), T([1536], bf16), T([1536], bf16), S([4096,1536]), S([4096,1536]), S([4096,1536]))
@oracle_impl(hardware="B200", point="20cc819f", BLOCK_M=1, BLOCK_H=2048, num_warps=8, num_stages=3)
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
        position_table,
        position_ids,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = inputs

    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(weight.shape[0])
    rows = batch * seq_len
    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )

    _deberta_embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        word_table,
        word_ids,
        position_table,
        position_ids,
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
        out.view(_as_shape(shape0)),
        out.view(_as_shape(shape1)),
        out.view(_as_shape(shape2)),
    )
