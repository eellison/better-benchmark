"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 residual-add LayerNorm alias scope with one fixed-hidden Triton row kernel, preserving the bf16 add boundary, fp32 population var_mean with eps 1e-5, affine epilogue, final bf16 cast, and four aliasing output views, whereas Inductor lowers the same graph through a generic var_mean normalization schedule; Inductor cannot do this today because correction=0 LayerNorm is not canonicalized to a direct fixed-width mean and centered-variance row template with alias-view emission; the fix is ALGEBRAIC_ELIMINATION: add a guarded residual-add LayerNorm lowering that replaces generic var_mean bookkeeping with direct row statistics and writes all alias views from one output buffer."""

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
def _round_bf16(x):
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;",
        constraints="=h,f",
        args=[x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _residual_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = _round_bf16(_f32_add(flat, residual)).to(tl.float32)

    x_reduce = tl.where(mask, x, 0.0)
    mean = (tl.sum(x_reduce, axis=1) * (1.0 / HIDDEN))[:, None]
    centered = _f32_sub(x, mean)
    centered_reduce = tl.where(mask, centered, 0.0)
    var = (tl.sum(_f32_mul(centered_reduce, centered_reduce), axis=1) * (1.0 / HIDDEN))[:, None]
    invstd = libdevice.rsqrt(_f32_add(var, EPS_VALUE))

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight[None, :])
    out = _round_bf16(_f32_add(scaled, bias[None, :]))
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="0b3dc49f", ROWS=8192, BATCH=8, SEQ=1024, HIDDEN=1024, BLOCK_H=1024, ROW_BLOCK=1, num_warps=4)
@oracle_impl(hardware="B200", point="d1f40ce2", ROWS=16384, BATCH=16, SEQ=1024, HIDDEN=768, BLOCK_H=1024, ROW_BLOCK=1, num_warps=4)
@oracle_impl(hardware="B200", point="d4cc3e3e", ROWS=16384, BATCH=64, SEQ=256, HIDDEN=1024, BLOCK_H=1024, ROW_BLOCK=1, num_warps=4)
def oracle_forward(inputs, *, ROWS, BATCH, SEQ, HIDDEN, BLOCK_H, ROW_BLOCK, num_warps):
    flat, residual, weight, bias = inputs[:4]
    base = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=flat.device,
        dtype=torch.bfloat16,
    )
    _residual_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        flat,
        residual,
        weight,
        bias,
        base,
        ROWS=ROWS,
        HIDDEN=HIDDEN,
        EPS_VALUE=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    view = base.view(ROWS, HIDDEN)
    return base, view, view, view
