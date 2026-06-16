"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Whisper bf16 residual-add LayerNorm scope in one Triton row kernel, including the metadata view of the flat input, the strided residual add as the normalization source, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt affine epilogue, final bf16 cast, the returned `[1,1500,384]` base tensor, and all eight returned `[1500,384]` view aliases from the same output storage, whereas Inductor lowers the strided add producer, clone/cast normalization source, row reduction, affine, and repeated alias-only view returns through its generic normalization scheduling path; Inductor cannot do this today because the norm-template scheduler does not keep the residual producer live while feeding the fixed-hidden LayerNorm template and preserving the repeated output-storage aliases; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to fuse strided residual-add producers, row statistics, affine bf16 stores, and repeated final-view aliases into one full-scope row plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _whisper_residual_layernorm_alias_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    RESIDUAL_STRIDE_B: tl.constexpr,
    RESIDUAL_STRIDE_S: tl.constexpr,
    RESIDUAL_STRIDE_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids[:, None] < ROWS
    col_mask = cols[None, :] < HIDDEN
    mask = row_mask & col_mask

    batch_ids = row_ids // SEQ_LEN
    seq_ids = row_ids - batch_ids * SEQ_LEN
    flat_offsets = row_ids[:, None] * HIDDEN + cols[None, :]
    residual_offsets = (
        batch_ids[:, None] * RESIDUAL_STRIDE_B
        + seq_ids[:, None] * RESIDUAL_STRIDE_S
        + cols[None, :] * RESIDUAL_STRIDE_H
    )

    flat = tl.load(flat_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + residual_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(residual, flat)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    variance = (
        tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1)[:, None]
        / HIDDEN
    )
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])
    tl.store(norm_out_ptr + flat_offsets, affine.to(tl.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# aafbb27e: (T([1500,384], bf16), T([1,1500,384], bf16, stride=(576000,1,1500)), T([384], bf16), T([384], bf16), S([1,1500,384]), S([1500,384]) x8)
@oracle_impl(hardware="B200", point="aafbb27e", BLOCK_H=512, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK, num_warps, num_stages):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
        shape7,
        shape8,
    ) = inputs
    base_shape = _as_shape(shape0)
    view_shape_1 = _as_shape(shape1)
    view_shape_2 = _as_shape(shape2)
    view_shape_3 = _as_shape(shape3)
    view_shape_4 = _as_shape(shape4)
    view_shape_5 = _as_shape(shape5)
    view_shape_6 = _as_shape(shape6)
    view_shape_7 = _as_shape(shape7)
    view_shape_8 = _as_shape(shape8)
    rows = int(view_shape_1[0])
    seq_len = int(base_shape[1])
    hidden = int(base_shape[2])

    norm_base = torch.empty_strided(
        base_shape,
        (seq_len * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _whisper_residual_layernorm_alias_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        norm_base,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        RESIDUAL_STRIDE_B=int(arg1_1.stride(0)),
        RESIDUAL_STRIDE_S=int(arg1_1.stride(1)),
        RESIDUAL_STRIDE_H=int(arg1_1.stride(2)),
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        norm_base,
        norm_base.view(view_shape_1),
        norm_base.view(view_shape_2),
        norm_base.view(view_shape_3),
        norm_base.view(view_shape_4),
        norm_base.view(view_shape_5),
        norm_base.view(view_shape_6),
        norm_base.view(view_shape_7),
        norm_base.view(view_shape_8),
    )
