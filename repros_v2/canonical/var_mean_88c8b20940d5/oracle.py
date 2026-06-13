"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BEiT class-token plus channels-last patch-token LayerNorm training scope in one Triton row kernel, including the `[128,768,14,14]` bf16 view/permute patch gather, f32 class-token expand, logical f32 `cat` materialization, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, returned f32 mean and rsqrt side tensors, affine scale/bias epilogue, final bf16 cast, and flattened `[25216,768]` view output, whereas Inductor lowers the expand/view/permute/cat producer, row statistics, side-output stores, affine cast, and final view through generic layout and normalization schedules; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not sink the class-token and channels-last patch gathers into the row-statistics load plan while also emitting the visible pre-norm cat, mean, rsqrt, and final bf16 view; the fix is SCHEDULER_FUSION: extend LayerNorm scheduling to fold static class-token/patch-token gathers into one row plan and emit all sibling outputs directly."""

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
def _beit_cat_layernorm_kernel(
    patch_ptr,
    class_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,
    mean_ptr,
    rsqrt_ptr,
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
    row_mask = row_offsets < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask
    offsets = rows * HIDDEN + cols

    batch = row_offsets[:, None] // TOKENS
    token = row_offsets[:, None] - batch * TOKENS
    patch = token - 1
    is_class = token == 0

    class_values = tl.load(
        class_ptr + cols,
        mask=col_mask & is_class,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    patch_values = tl.load(
        patch_ptr
        + batch * PATCH_STRIDE_B
        + patch * PATCH_STRIDE_P
        + cols * PATCH_STRIDE_H,
        mask=mask & (token != 0) & (patch < PATCHES),
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = tl.where(is_class, class_values, patch_values)
    x = tl.where(mask, x, 0.0)
    tl.store(cat_ptr + offsets, x, mask=mask)

    mean = _f32_mul(tl.sum(x, axis=1), 1.0 / HIDDEN)
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

    tl.store(mean_ptr + row_offsets, mean, mask=row_mask)
    tl.store(rsqrt_ptr + row_offsets, invstd, mask=row_mask)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="304d27fb", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, shape2 = inputs
    batch = int(arg0_1.shape[0])
    hidden = int(arg2_1.shape[0])
    patch_count = int(arg0_1.shape[2]) * int(arg0_1.shape[3])
    tokens = patch_count + 1
    rows = batch * tokens
    cat_shape = (batch, tokens, hidden)
    side_shape = (batch, tokens, 1)
    out_shape = _as_shape(shape2)

    cat = torch.empty_strided(
        cat_shape,
        _contiguous_stride(cat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        side_shape,
        _contiguous_stride(side_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        side_shape,
        _contiguous_stride(side_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _beit_cat_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        cat,
        mean,
        rsqrt,
        out,
        ROWS=rows,
        TOKENS=tokens,
        PATCHES=patch_count,
        HIDDEN=hidden,
        PATCH_STRIDE_B=arg0_1.stride(0),
        PATCH_STRIDE_P=arg0_1.stride(3),
        PATCH_STRIDE_H=arg0_1.stride(1),
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return cat, mean, rsqrt, out
