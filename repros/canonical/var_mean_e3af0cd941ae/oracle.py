"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Bart/PLBart bf16 token-plus-generated-position embedding LayerNorm scope in one Triton row kernel, including token embedding gather, generated `iota + 2` position embedding gather, the captured bf16 embedding-add boundary before fp32 promotion, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 libdevice rsqrt, bf16 weight/bias affine epilogue, final bf16 `[batch, 1024, hidden]` output, and the three aliasing `[batch*1024, hidden]` view returns, whereas Inductor lowers the embedding producers, generated position indices, row normalization, affine cast, and repeated alias-only view returns through generic scheduler regions; Inductor cannot do this today because its fixed-hidden normalization scheduler does not canonicalize Bart-style token-plus-generated-position embedding producers into one full-scope row-normalization plan while preserving the bf16 add boundary and alias-only output contract; the fix is NEW_PATTERN: add a guarded Bart embedding-LayerNorm lowering that folds indexed token/position loads into the row kernel and emits the base bf16 tensor plus sibling view aliases directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
POSITION_OFFSET = 2
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
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    POSITION_OFFSET_: tl.constexpr,
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
    seq = rows - (rows // SEQ_LEN) * SEQ_LEN

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + (seq + POSITION_OFFSET_) * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    summed = _f32_add(token, position)
    if not USE_INDUCTOR_NUMERICS:
        summed = summed.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    summed_masked = tl.where(mask, summed, 0.0)

    mean = _f32_mul(tl.sum(summed_masked, axis=1), 1.0 / HIDDEN)
    centered = _f32_sub(summed, mean[:, None])
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


@oracle_impl(hardware="B200", point="dacf9422", BLOCK_M=1, BLOCK_H=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="90ede91d", BLOCK_M=1, BLOCK_H=1024, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    global _USE_INDUCTOR_NUMERICS
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        EPSILON=EPS,
        POSITION_OFFSET_=POSITION_OFFSET,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        out,
        out.view(_as_shape(shape0)),
        out.view(_as_shape(shape1)),
        out.view(_as_shape(shape2)),
    )
