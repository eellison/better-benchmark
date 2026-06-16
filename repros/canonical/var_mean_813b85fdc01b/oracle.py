"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete returned DeiT bf16 residual-add LayerNorm scope in one fixed-hidden Triton row kernel by sinking select(..., 1, 0).clone() through the row-independent view, residual add, fp32 population var_mean(..., dim=2, correction=0, keepdim=True), eps=1e-6 rsqrt, affine epilogue, and final bf16 cast, so only the live class-token rows are reduced and stored, whereas Inductor lowers the generic normalization producer over all [128,197] token rows before selecting token zero; Inductor cannot do this today because norm-template canonicalization does not narrow a row-wise normalization domain from a downstream constant token select while preserving the add and cast boundaries; the fix is ALGEBRAIC_ELIMINATION: commute constant token selects through row-local normalization and eliminate dead patch-token rows before scheduling the LayerNorm template."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
HIDDEN = 192
EPS = 1.0e-6


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
def _class_token_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BATCH_C: tl.constexpr,
    TOKENS_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batches = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    cols = tl.arange(0, BLOCK_H)
    batch_mask = batches < BATCH_C
    col_mask = cols < HIDDEN_C
    mask = batch_mask[:, None] & col_mask[None, :]

    input_offsets = batches[:, None] * TOKENS_C * HIDDEN_C + cols[None, :]
    flat = tl.load(
        flat_ptr + input_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + input_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    x = _f32_add(residual, flat)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN_C
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN_C
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
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
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])

    out_offsets = batches[:, None] * HIDDEN_C + cols[None, :]
    tl.store(
        out_ptr + out_offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


# ad6d6241: (T([25216,192], bf16), T([128,197,192], bf16), T([192], bf16), T([192], bf16), S([128,197,192]))
@oracle_impl(hardware="B200", point="ad6d6241", BLOCK_B=4, BLOCK_H=256, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_B: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    flat, residual, weight, bias, _shape0 = inputs
    out = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=flat.device,
        dtype=torch.bfloat16,
    )
    _class_token_layernorm_kernel[(triton.cdiv(BATCH, BLOCK_B),)](
        flat,
        residual,
        weight,
        bias,
        out,
        BATCH_C=BATCH,
        TOKENS_C=TOKENS,
        HIDDEN_C=HIDDEN,
        EPS_C=EPS,
        BLOCK_B=BLOCK_B,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
