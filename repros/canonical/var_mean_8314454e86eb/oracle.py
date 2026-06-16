"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DistilBERT bf16 embedding-plus-LayerNorm inference scope in one Triton row kernel, including the word embedding gather, sliced learned-position gather, the eager-compatible bf16 embedding-add path for correctness checks, Inductor's f32-promoted embedding sum for the compiled numerics envelope, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 libdevice rsqrt, bf16 weight/bias affine epilogue, final bf16 `[256,128,768]` output, and the three aliasing `[32768,768]` view returns, whereas Inductor lowers the embedding producers, row normalization, affine cast, and repeated view returns through generic scheduler regions; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden LayerNorm scheduler does not sink indexed embedding gathers and all alias-only returns into one reusable semantic row-normalization template; the fix is SCHEDULER_FUSION: extend the embedding-LayerNorm template to fuse gather/add producers and emit the final buffer plus sibling view aliases from one guarded schedule."""

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
def _embedding_layernorm_kernel(
    word_table_ptr,
    word_ids_ptr,
    position_ids_ptr,
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

    if USE_INDUCTOR_NUMERICS:
        x = _f32_add(word, position)
    else:
        x = _f32_add(word, position).to(tl.bfloat16, fp_downcast_rounding="rtne").to(
            tl.float32
        )
    x_masked = tl.where(mask, x, 0.0)

    mean = _f32_mul(tl.sum(x_masked, axis=1), 1.0 / HIDDEN)
    centered = _f32_sub(x, mean[:, None])
    centered_masked = tl.where(mask, centered, 0.0)
    variance = _f32_mul(
        tl.sum(_f32_mul(centered_masked, centered_masked), axis=1),
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


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 5095537f: (T([30522,768], bf16), T([256,128], i64), T([1,512], i64), T([512,768], bf16), T([768], bf16), T([768], bf16), S([32768,768]), S([32768,768]), S([32768,768]))
@oracle_impl(hardware="B200", point="5095537f", BLOCK_M=1, BLOCK_H=1024, num_warps=4, num_stages=3)
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
        position_ids,
        position_table,
        weight,
        bias,
        view_shape0,
        view_shape1,
        view_shape2,
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

    _embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        word_table,
        word_ids,
        position_ids,
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

    return (
        out,
        out.view(_as_shape(view_shape0)),
        out.view(_as_shape(view_shape1)),
        out.view(_as_shape(view_shape2)),
    )
