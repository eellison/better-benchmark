"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeiT distilled-token patch LayerNorm inference scope in one Triton row kernel, including both expanded bf16 token rows, channels-last `[128,768,14,14]` patch gather through the view/permute, positional add materialized as the returned bf16 `[128,198,768]` tensor, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, fp32 affine epilogue with bf16 weight and bias, final bf16 cast, and the flattened `[25344,768]` view output; Inductor lowers this through generic cat/add and normalization schedules and cannot currently fuse the static token/patch/position gather with row-statistics reduction while also emitting the returned pre-norm add buffer, so the fix is SCHEDULER_FUSION: extend fixed-hidden LayerNorm scheduling to fold these static DeiT token producers into one row plan with both required outputs."""

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
def _deit_distilled_patch_layernorm_kernel(
    class_token_ptr,
    dist_token_ptr,
    patch_ptr,
    pos_ptr,
    weight_ptr,
    bias_ptr,
    add_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    PATCHES: tl.constexpr,
    HIDDEN: tl.constexpr,
    PATCH_STRIDE_B: tl.constexpr,
    PATCH_STRIDE_P: tl.constexpr,
    PATCH_STRIDE_H: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (rows < ROWS) & (cols < HIDDEN)
    col_mask = cols < HIDDEN

    batch = row_offsets[:, None] // TOKENS
    token = row_offsets[:, None] - batch * TOKENS
    patch = token - 2
    offsets = rows * HIDDEN + cols

    class_values = tl.load(
        class_token_ptr + cols,
        mask=col_mask & (token == 0),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.bfloat16)
    dist_values = tl.load(
        dist_token_ptr + cols,
        mask=col_mask & (token == 1),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.bfloat16)
    patch_values = tl.load(
        patch_ptr + batch * PATCH_STRIDE_B + patch * PATCH_STRIDE_P + cols * PATCH_STRIDE_H,
        mask=mask & (token >= 2) & (patch < PATCHES),
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    pos_values = tl.load(
        pos_ptr + token * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.bfloat16)

    cat_values = tl.where(token == 0, class_values, tl.where(token == 1, dist_values, patch_values))
    add_values = _f32_add(cat_values.to(tl.float32), pos_values.to(tl.float32)).to(tl.bfloat16)
    tl.store(add_ptr + offsets, add_values, mask=mask)

    x = add_values.to(tl.float32)
    reduce_input = tl.where(mask, x, 0.0)
    mean = _f32_mul(tl.sum(reduce_input, axis=1), 1.0 / HIDDEN)
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
        out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="03ce16f1", BLOCK_H=1024, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (
        class_token,
        dist_token,
        patches,
        pos_embed,
        weight,
        bias,
        _class_expand_shape,
        _dist_expand_shape,
        _patch_view_shape,
        output_view_shape,
    ) = inputs

    batch = int(patches.shape[0])
    hidden = int(weight.shape[0])
    patch_count = int(patches.shape[2]) * int(patches.shape[3])
    tokens = patch_count + 2
    rows = batch * tokens
    base_shape = (batch, tokens, hidden)
    base_stride = _contiguous_stride(base_shape)

    add = torch.empty_strided(
        base_shape,
        base_stride,
        device=patches.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        base_shape,
        base_stride,
        device=patches.device,
        dtype=torch.bfloat16,
    )

    _deit_distilled_patch_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        class_token,
        dist_token,
        patches,
        pos_embed,
        weight,
        bias,
        add,
        out,
        ROWS=rows,
        TOKENS=tokens,
        PATCHES=patch_count,
        HIDDEN=hidden,
        PATCH_STRIDE_B=patches.stride(0),
        PATCH_STRIDE_P=patches.stride(3),
        PATCH_STRIDE_H=patches.stride(1),
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return add, out.view(_as_shape(output_view_shape))
