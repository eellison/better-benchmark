"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete M2M100 bf16 scaled-token embedding, computed-position embedding gather, observable bf16 embedding-add output, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` over that rounded add, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and three aliasing `[8192,1024]` view returns in one Triton row kernel, whereas Inductor materializes the indexed embedding/add region before lowering the fixed-hidden normalization and repeated view returns through generic scheduler regions; Inductor cannot do this today because its normalization scheduler does not sink row-local token gathers, computed position-index arithmetic, and the bf16 add boundary into the hidden-size-1024 reduction template while preserving alias-only sibling outputs; the fix is SCHEDULER_FUSION: extend fixed-hidden LayerNorm codegen to fuse safe rowwise gather/add producers with scalar index arithmetic and emit the observable add plus final view aliases directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
EMBED_SCALE = 32.0
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
def _m2m100_embedding_layernorm_kernel(
    token_table_ptr,
    token_ids_ptr,
    positions_ptr,
    position_mask_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    final_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    EMBED_SCALE_: tl.constexpr,
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

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
    pos_i32 = tl.load(positions_ptr + rows, mask=row_mask, other=0).to(tl.int32)
    pos_mask = tl.load(position_mask_ptr + rows, mask=row_mask, other=0).to(tl.int32)
    position_id = (pos_i32 * pos_mask).to(tl.int64) + 1

    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols,
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

    scaled = _f32_mul(token, EMBED_SCALE_)
    add_f32 = _f32_add(
        scaled.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
        position,
    )
    add_bf16 = add_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    x = add_bf16.to(tl.float32)
    if USE_INDUCTOR_NUMERICS:
        x = _f32_add(scaled, position)
    x_masked = tl.where(mask, x, 0.0)
    mean = _f32_mul(tl.sum(x_masked, axis=1), 1.0 / HIDDEN)
    centered = _f32_sub(x, mean[:, None])
    centered_masked = tl.where(mask, centered, 0.0)
    variance = _f32_mul(
        tl.sum(_f32_mul(centered_masked, centered_masked), axis=1),
        1.0 / HIDDEN,
    )
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
        final_out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 929234c0: (T([128112,1024], bf16), T([64,128], i64), T([64,128], i64), T([64,128], i32), T([1026,1024], bf16), T([1024], bf16), T([1024], bf16), S([64,128,1024]), S([8192,1024]), S([8192,1024]), S([8192,1024]))
@oracle_impl(hardware="B200", point="929234c0", BLOCK_M=1, BLOCK_H=1024, num_warps=4, num_stages=3)
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
        positions,
        position_mask,
        position_table,
        weight,
        bias,
        add_shape,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len
    base_shape = _as_shape(add_shape)
    base_stride = (seq_len * hidden, hidden, 1)
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    add_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _m2m100_embedding_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        token_table,
        token_ids,
        positions,
        position_mask,
        position_table,
        weight,
        bias,
        add_out,
        final_out,
        ROWS=rows,
        HIDDEN=hidden,
        EPSILON=EPS,
        EMBED_SCALE_=EMBED_SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        add_out,
        final_out.view(_as_shape(view_shape0)),
        final_out.view(_as_shape(view_shape1)),
        final_out.view(_as_shape(view_shape2)),
    )
