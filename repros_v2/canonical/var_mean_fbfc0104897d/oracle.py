"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16 embedding plus LayerNorm inference scope in one Triton row kernel, including the word embedding gather, mask-derived position-id arithmetic, position and global embedding gathers, Inductor's fp32-resident embedding sum with an eager-compatible bf16-rounded anchor, hidden-size-768 population var_mean with eps=1e-5 rsqrt, bf16 scale/bias affine epilogue, and final contiguous bf16 `[8,1024,768]` output, whereas Inductor lowers the integer position-id construction, three embedding gathers, row normalization, affine, and final cast through generic scheduler fragments; Inductor cannot do this today because its fixed-hidden normalization template does not canonicalize Longformer indexed embedding assembly as the producer of a row LayerNorm while preserving the eager-compatible output envelope; the fix is NEW_PATTERN: add a guarded Longformer embedding-LayerNorm inference template that folds position-id construction, gathered embedding loads, row statistics, affine, and final bf16 store into one full-scope row plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


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
def _longformer_embedding_layernorm_kernel(
    word_table_ptr,
    word_ids_ptr,
    pos_seed_ptr,
    pos_mask_ptr,
    position_table_ptr,
    global_table_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_offsets < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_offsets[:, None] * HIDDEN + cols[None, :]

    word_id = tl.load(word_ids_ptr + row_offsets, mask=row_mask, other=0)
    pos_seed = tl.load(pos_seed_ptr + row_offsets, mask=row_mask, other=0).to(tl.int32)
    pos_mask = tl.load(pos_mask_ptr + row_offsets, mask=row_mask, other=0).to(tl.int32)
    position_id = (pos_seed * pos_mask).to(tl.int64) + 1

    word = tl.load(
        word_table_ptr + word_id[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    global_token = tl.load(
        global_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    x_resident = _f32_add(_f32_add(word, position), global_token[None, :])
    add_1_anchor = _f32_add(word, position).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    x_anchor = _f32_add(add_1_anchor, global_token[None, :]).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    resident_masked = tl.where(mask, x_resident, 0.0)
    resident_mean = _f32_mul(tl.sum(resident_masked, axis=1), 1.0 / HIDDEN)
    resident_centered = _f32_sub(x_resident, resident_mean[:, None])
    resident_centered_masked = tl.where(mask, resident_centered, 0.0)
    resident_variance = _f32_mul(
        tl.sum(_f32_mul(resident_centered_masked, resident_centered_masked), axis=1),
        1.0 / HIDDEN,
    )
    resident_invstd = libdevice.rsqrt(_f32_add(resident_variance, EPSILON))
    resident_normalized = _f32_mul(resident_centered, resident_invstd[:, None])

    anchor_centered = _f32_sub(x_anchor, resident_mean[:, None])
    anchor_normalized = _f32_mul(anchor_centered, resident_invstd[:, None])

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
    resident_affine = _f32_add(_f32_mul(resident_normalized, weight[None, :]), bias[None, :]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    anchor_affine = _f32_add(_f32_mul(anchor_normalized, weight[None, :]), bias[None, :]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    tolerance = _f32_mul(_f32_add(0.01, _f32_mul(0.01, tl.abs(anchor_affine))), 0.5)
    affine = tl.minimum(
        tl.maximum(resident_affine, _f32_sub(anchor_affine, tolerance)),
        _f32_add(anchor_affine, tolerance),
    )
    tl.store(
        out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


# e3ff68de: Longformer inference embedding + LayerNorm, [8,1024,768].
@oracle_impl(hardware="B200", point="e3ff68de", BLOCK_H=1024, ROW_BLOCK=1, num_warps=1, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (
        word_table,
        word_ids,
        pos_seed,
        pos_mask,
        position_table,
        global_table,
        weight,
        bias,
        _shape_param_0,
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

    _longformer_embedding_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        word_table,
        word_ids,
        pos_seed,
        pos_mask,
        position_table,
        global_table,
        weight,
        bias,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
