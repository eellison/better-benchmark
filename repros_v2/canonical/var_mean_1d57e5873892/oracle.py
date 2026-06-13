"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BEiT bf16 class-token plus patch-token LayerNorm inference scope in one Triton row kernel, including class-token expansion, channels-last patch view/permute gather from `[128,768,14,14]`, the returned bf16 `[128,197,768]` cat materialization, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, fp32 affine epilogue with bf16 weight and bias, final bf16 cast, and the flattened `[25216,768]` view output, whereas Inductor lowers the expand/view/permute/cat producer, normalization reduction, affine cast, and view return through generic layout and normalization schedules; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not sink class-token and channels-last patch gathers into the row-statistics load plan while also preserving the returned bf16 cat buffer; the fix is SCHEDULER_FUSION: extend the LayerNorm scheduler to fold static class-token/patch-token gathers into one row plan and emit both the pre-norm cat tensor and final flattened affine output directly."""

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
def _cat_patch_layernorm_kernel(
    class_token_ptr,
    patch_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,
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
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN

    batch = row_offsets[:, None] // TOKENS
    token = row_offsets[:, None] - batch * TOKENS
    patch = token - 1
    is_class = token == 0
    offsets = rows * HIDDEN + cols

    class_values = tl.load(
        class_token_ptr + cols,
        mask=col_mask & is_class,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.bfloat16)
    patch_values = tl.load(
        patch_ptr + batch * PATCH_STRIDE_B + patch * PATCH_STRIDE_P + cols * PATCH_STRIDE_H,
        mask=mask & (token != 0) & (patch < PATCHES),
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    cat_values = tl.where(is_class, class_values, patch_values).to(tl.bfloat16)
    tl.store(cat_ptr + offsets, cat_values, mask=mask)

    x = cat_values.to(tl.float32)
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
        mask=mask & row_mask,
    )


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 8b5672b4: (T([1,1,768], bf16), T([128,768,14,14], bf16, stride=(150528,1,10752,768)), T([768], bf16), T([768], bf16), ...)
@oracle_impl(hardware="B200", point="8b5672b4", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    class_token, patches, weight, bias, _expand_shape, _patch_shape, view_shape = inputs
    batch = int(patches.shape[0])
    hidden = int(weight.shape[0])
    patch_count = int(patches.shape[2]) * int(patches.shape[3])
    tokens = patch_count + 1
    rows = batch * tokens
    cat_shape = (batch, tokens, hidden)

    cat = torch.empty_strided(
        cat_shape,
        (tokens * hidden, hidden, 1),
        device=patches.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        cat_shape,
        (tokens * hidden, hidden, 1),
        device=patches.device,
        dtype=torch.bfloat16,
    )

    _cat_patch_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        class_token,
        patches,
        weight,
        bias,
        cat,
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

    return cat, out.view(_as_shape(view_shape))
