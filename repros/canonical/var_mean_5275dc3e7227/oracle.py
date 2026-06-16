"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DeiT bf16-residual/f32-activation LayerNorm scope, including the view, residual add, fp32 population var_mean over hidden size 768, eps-before-libdevice.rsqrt normalization, the full returned normalized f32 tensor, `rsqrt / 768`, and bf16 affine stores only for selected tokens 0 and 1, whereas Inductor currently treats the affine epilogue as live for every `[128,198,768]` row before selecting two tokens; Inductor cannot do this today because its normalization scheduler does not propagate fixed token selects backward through the affine epilogue while preserving the all-row normalized tensor and inverse-std side output; the fix is ALGEBRAIC_ELIMINATION: sink constant token selects into the LayerNorm epilogue and eliminate scale/bias work and stores for tokens 2..197."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _layernorm_normalized_selected_affine_kernel(
    residual_bf16_ptr,
    activation_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    selected0_ptr,
    selected1_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    activation = tl.load(
        activation_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_bf16_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(activation, residual)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    variance_sum = tl.sum(
        tl.where(mask, _f32_mul(centered, centered), 0.0),
        axis=1,
    )
    variance = variance_sum / HIDDEN
    eps = tl.full((ROW_BLOCK,), EPSILON, tl.float32)
    invstd = libdevice.rsqrt(_f32_add(variance, eps))
    normalized = _f32_mul(centered, invstd[:, None])

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    hidden_value = tl.full((ROW_BLOCK,), HIDDEN, tl.float32)
    tl.store(div_ptr + row_ids, _f32_div(invstd, hidden_value), mask=row_mask)

    token = row_ids - (row_ids // TOKENS) * TOKENS
    selected = token < 2
    selected_mask = selected[:, None] & col_mask[None, :]
    weight = tl.load(
        weight_ptr + cols,
        mask=selected_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=selected_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    batch = row_ids // TOKENS
    selected_offsets = batch[:, None] * HIDDEN + cols[None, :]
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(
        selected0_ptr + selected_offsets,
        affine_bf16,
        mask=(token[:, None] == 0) & col_mask[None, :],
    )
    tl.store(
        selected1_ptr + selected_offsets,
        affine_bf16,
        mask=(token[:, None] == 1) & col_mask[None, :],
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# f13eb73e: (T([25344,768], bf16), T([128,198,768], f32), T([768], f32), T([768], f32), S([128,198,768]))
@oracle_impl(hardware="B200", point="f13eb73e", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    residual_bf16, activation, weight, bias, shape0 = inputs
    norm_shape = _shape_tuple(shape0)
    batch = int(activation.shape[0])
    tokens = int(activation.shape[1])
    hidden = int(activation.shape[2])
    rows = int(residual_bf16.shape[0])

    normalized = torch.empty_strided(
        norm_shape,
        (tokens * hidden, hidden, 1),
        device=activation.device,
        dtype=torch.float32,
    )
    selected0 = torch.empty_strided(
        (batch, hidden),
        (hidden, 1),
        device=activation.device,
        dtype=torch.bfloat16,
    )
    selected1 = torch.empty_strided(
        (batch, hidden),
        (hidden, 1),
        device=activation.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        (batch, tokens, 1),
        (tokens, 1, 1),
        device=activation.device,
        dtype=torch.float32,
    )

    _layernorm_normalized_selected_affine_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        residual_bf16,
        activation,
        weight,
        bias,
        normalized,
        selected0,
        selected1,
        div,
        ROWS=rows,
        TOKENS=tokens,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return normalized, selected0, selected1, div
