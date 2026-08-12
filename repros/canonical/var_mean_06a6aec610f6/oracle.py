"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the faithful DINOv2 class-token LayerNorm return scope by sinking the trailing `select(dim=1, index=0).clone()` through the row-independent bf16 scaled residual producer, fp32 population `var_mean(..., correction=0)`, eps=1e-6 rsqrt, affine scale/bias, and final bf16 cast, so only the 128 observable token-0 rows are reduced and stored, whereas Inductor lowers the captured graph as a full `[128,1370,768]` normalization and affine producer before applying the class-token select; Inductor cannot do this today because its normalization scheduler does not commute a constant token select backward through row-local LayerNorm and final cast to prove the other 1369 token rows per batch are dead; the fix is ALGEBRAIC_ELIMINATION: push fixed-token selects through row-local normalization graphs and narrow the scheduled row domain before codegen while preserving the returned bf16 output scope."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
HIDDEN = 768
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
    source_ptr,
    scale_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOKENS_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_C
    src_row = batch * TOKENS_C
    src_offsets = src_row * HIDDEN_C + cols

    source = tl.load(
        source_ptr + src_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + src_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    weight = tl.load(
        weight_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    scaled = _f32_mul(source, scale).to(tl.bfloat16).to(tl.float32)
    x = _f32_add(residual, scaled).to(tl.bfloat16).to(tl.float32)
    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=0) / HIDDEN_C
    centered = _f32_sub(x, mean)
    variance = tl.sum(
        tl.where(mask, _f32_mul(centered, centered), 0.0),
        axis=0,
    ) / HIDDEN_C
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    norm = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(norm, weight), bias)
    out = affine.to(tl.bfloat16)
    tl.store(out_ptr + batch * HIDDEN_C + cols, out, mask=mask)


# 01a8f9c7: (T([175360,768], bf16), T([768], bf16), T([128,1370,768], bf16), T([768], bf16), T([768], bf16), S([128,1370,768]))
@oracle_impl(hardware="B200", point="01a8f9c7", BLOCK_H=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    source, scale, residual, weight, bias, _shape0 = inputs
    out = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=source.device,
        dtype=torch.bfloat16,
    )
    _class_token_layernorm_kernel[(BATCH,)](
        source,
        scale,
        residual,
        weight,
        bias,
        out,
        TOKENS_C=TOKENS,
        HIDDEN_C=HIDDEN,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
