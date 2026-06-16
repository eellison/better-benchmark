"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete YituTechConvBert bf16 embedding-plus-LayerNorm inference scope in one Triton row kernel, including the word embedding gather, learned-position gather, token-type gather through the captured expand indices, the eager-compatible bf16 embedding-add path for correctness checks, Inductor's f32-promoted embedding sum for the compiled numerics envelope, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, bf16 weight/bias affine epilogue, final bf16 `[32,512,768]` output, four aliasing `[16384,768]` view returns, and the `[32,768,512]` permute alias, whereas Inductor lowers the embedding producers, row normalization, affine cast, repeated views, and permute alias through a generic persistent reduction region; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden LayerNorm scheduler does not sink indexed embedding gathers and all alias-only returns into one reusable semantic row-normalization template; the fix is SCHEDULER_FUSION: extend the embedding-LayerNorm template to fuse gather/add producers and emit the final buffer plus flat-view and permute aliases from one guarded schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-12
_USE_INDUCTOR_NUMERICS = False


@triton.jit
def _embedding_layernorm_kernel(
    word_table_ptr,
    word_ids_ptr,
    position_table_ptr,
    position_ids_ptr,
    token_type_ids_ptr,
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
    token_type_id = tl.load(token_type_ids_ptr + seq, mask=row_mask, other=0)

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
    token_type = tl.load(
        token_type_table_ptr + token_type_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    if USE_INDUCTOR_NUMERICS:
        x = word + position
        x = x + token_type
    else:
        add0 = (word + position).to(tl.bfloat16).to(tl.float32)
        x = (add0 + token_type).to(tl.bfloat16).to(tl.float32)

    mean = tl.sum(tl.where(col_mask, x, 0.0), axis=1) / HIDDEN
    centered = x - mean[:, None]
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(centered_for_var * centered_for_var, axis=1) / HIDDEN
    invstd = libdevice.rsqrt(variance + EPSILON)
    normalized = centered * invstd[:, None]

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
    affine = normalized * weight + bias
    tl.store(
        out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a6271911", BLOCK_M=2, BLOCK_H=1024, num_warps=8, num_stages=3)
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
        token_type_ids,
        token_type_table,
        weight,
        bias,
        _expand_shape,
        view_shape0,
        view_shape1,
        view_shape2,
        view_shape3,
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
        position_table,
        position_ids,
        token_type_ids,
        token_type_table,
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
        out.view(_as_shape(view_shape3)),
        out.permute(0, 2, 1),
    )
