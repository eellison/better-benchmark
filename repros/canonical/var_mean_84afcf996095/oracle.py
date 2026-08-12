"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SigLIP ViT channels-last patch-token residual LayerNorm inference scope in one Triton row kernel, including the `[128,768,16,16]` bf16 view/permute patch gather, broadcasted `[1,256,768]` bf16 positional add with returned bf16 add tensor, bf16-to-fp32 conversion before population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, bf16 affine weight/bias epilogue, final bf16 cast, and flattened `[32768,768]` view output, whereas Inductor lowers the patch view/permute producer, returned residual-add, row statistics, affine cast, and final view through generic layout and normalization schedules; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not sink the channels-last patch gather and broadcasted bf16 residual add into the row-statistics load plan while also emitting the visible rounded add tensor; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to fold static patch gathers and returned bf16 residual producers into one full-scope row plan."""

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
def _patch_residual_layernorm_kernel(
    patch_ptr,
    pos_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
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

    patch = tl.load(
        patch_ptr
        + batch * PATCH_STRIDE_B
        + token * PATCH_STRIDE_P
        + cols * PATCH_STRIDE_H,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    pos = tl.load(
        pos_ptr + token * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    )
    added = (patch + pos).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, added, mask=mask)

    x = added.to(tl.float32)
    x_for_reduce = tl.where(mask, x, 0.0)
    mean = _f32_mul(tl.sum(x_for_reduce, axis=1), 1.0 / HIDDEN)
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


@oracle_impl(hardware="B200", point="3eee3489", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    batch = int(view_shape[0])
    hidden = int(view_shape[1])
    tokens = int(view_shape[2])
    rows = batch * tokens
    add_shape = (batch, tokens, hidden)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _patch_residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        out,
        ROWS=rows,
        TOKENS=tokens,
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
    return add_out, out
