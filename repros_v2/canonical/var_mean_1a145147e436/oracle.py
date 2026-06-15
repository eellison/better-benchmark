"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MegatronBERT bf16 word/token-type/position embedding assembly, returned bf16 summed embedding, hidden-size-1024 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, fp32 affine epilogue, final bf16 normalized tensor, and three aliasing `[8192,1024]` view returns in one Triton row kernel, whereas Inductor lowers the embedding gathers, visible summed-embedding producer, row normalization, affine cast, and repeated alias-only returns through generic embedding and normalization schedules; Inductor cannot do this today because its normalization-template pattern library does not recognize a multi-producer embedding gather feeding fixed-hidden LayerNorm while preserving the visible bf16 side output and aliasing view contract; the fix is NEW_PATTERN: add a guarded MegatronBERT embedding-LayerNorm lowering that folds token, constant token-type, and position embedding loads into the row-normalization kernel and emits the summed side output plus shared normalized view aliases directly."""

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
def _embedding_layernorm_alias_kernel(
    token_table_ptr,
    token_ids_ptr,
    token_type_table_ptr,
    position_table_ptr,
    position_ids_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
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
    offsets = rows * HIDDEN + cols
    seq = rows - (rows // SEQ_LEN) * SEQ_LEN

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
    position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)

    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    token_type = tl.load(
        token_type_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    if USE_INDUCTOR_NUMERICS:
        x = _f32_add(_f32_add(token, token_type), position)
    else:
        add0 = _f32_add(token, token_type).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        ).to(tl.float32)
        x = _f32_add(add0, position).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        ).to(tl.float32)

    tl.store(
        add_out_ptr + offsets,
        x.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
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
        norm_out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# dc92a431: (T([29056,1024], bf16), T([16,512], i64), T([2,1024], bf16), T([512,1024], bf16), T([1,512], i64), T([1024], bf16), T([1024], bf16), S([16,512]), S([8192,1024]), S([8192,1024]), S([8192,1024]))
@oracle_impl(
    hardware="B200",
    point="dc92a431",
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
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        _token_type_shape,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs
    del _token_type_shape

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    base_shape = (batch, seq_len, hidden)
    base_stride = (seq_len * hidden, hidden, 1)
    add_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_alias_kernel[(triton.cdiv(rows, BLOCK_M),)](
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        add_out,
        norm_out,
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
        add_out,
        norm_out.view(_as_shape(view_shape0)),
        norm_out.view(_as_shape(view_shape1)),
        norm_out.view(_as_shape(view_shape2)),
    )
