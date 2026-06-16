"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Whisper bf16 residual-add LayerNorm scope in one Triton row kernel, including the metadata view of the flat input, the strided bf16 residual add returned with eager's `(576000, 1, 1500)` layout, the clone-equivalent contiguous fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and contiguous `[1500,384]` view return, whereas Inductor lowers the strided add producer, clone/cast normalization source, row reduction, affine, and visible side output through its generic normalization scheduling path; Inductor cannot do this today because the norm-template scheduler does not keep a non-contiguous returned bf16 residual producer live while also feeding its contiguous rounded values into the fixed-hidden LayerNorm template; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to fuse strided visible residual-add producers, clone-equivalent row statistics, and direct final-view stores into one full-scope row plan."""

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
def _whisper_returned_add_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    RESIDUAL_STRIDE_B: tl.constexpr,
    RESIDUAL_STRIDE_S: tl.constexpr,
    RESIDUAL_STRIDE_H: tl.constexpr,
    ADD_STRIDE_B: tl.constexpr,
    ADD_STRIDE_S: tl.constexpr,
    ADD_STRIDE_H: tl.constexpr,
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
    add_offsets = (
        batch_ids[:, None] * ADD_STRIDE_B
        + seq_ids[:, None] * ADD_STRIDE_S
        + cols[None, :] * ADD_STRIDE_H
    )

    flat = tl.load(flat_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + residual_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    added = _f32_add(residual, flat).to(tl.bfloat16)
    tl.store(add_out_ptr + add_offsets, added, mask=mask)

    x = added.to(tl.float32)
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


# aafbb27e: (T([1500,384], bf16), T([1,1500,384], bf16, stride=(576000,1,1500)), T([384], bf16), T([384], bf16), S([1,1500,384]), S([1500,384]))
@oracle_impl(hardware="B200", point="aafbb27e", BLOCK_H=512, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK, num_warps, num_stages):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    add_shape = tuple(int(dim) for dim in shape0)
    norm_shape = tuple(int(dim) for dim in shape1)
    rows = int(norm_shape[0])
    seq_len = int(add_shape[1])
    hidden = int(add_shape[2])

    add_out = torch.empty_strided(
        add_shape,
        (seq_len * hidden, 1, seq_len),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _whisper_returned_add_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        RESIDUAL_STRIDE_B=int(arg1_1.stride(0)),
        RESIDUAL_STRIDE_S=int(arg1_1.stride(1)),
        RESIDUAL_STRIDE_H=int(arg1_1.stride(2)),
        ADD_STRIDE_B=seq_len * hidden,
        ADD_STRIDE_S=1,
        ADD_STRIDE_H=seq_len,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
