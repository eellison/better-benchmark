"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin bf16 window-reverse, cyclic-shift residual add, and LayerNorm scope in one Triton row kernel, including the `[8192,49,128] -> [128,56,56,128]` window reverse, the iota-indexed `(idx + 53) % 56` shift on both spatial axes, the returned bf16 `[128,3136,128]` residual-add view, explicit bf16-to-fp32 conversion before population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, bf16 weight/bias affine epilogue in fp32, final bf16 cast, and returned `[401408,128]` view; Inductor lowers the window reverse, index shift, residual add, normalization, and final cast/view through generic layout and norm-template schedules; the fix is SCHEDULER_FUSION: extend fixed-hidden LayerNorm scheduling to inline deterministic Swin remaps and emit both the pre-norm bf16 view and final normalized bf16 view from one row tile."""

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
def _swin_shifted_reverse_residual_layernorm_kernel(
    window_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    HIDDEN: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCKS_H: tl.constexpr,
    BLOCKS_W: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask

    batch = row_ids // TOKENS
    spatial = row_ids - batch * TOKENS
    h = spatial // WIDTH
    w = spatial - h * WIDTH
    src_h = (h + SHIFT) % HEIGHT
    src_w = (w + SHIFT) % WIDTH
    block_h = src_h // WINDOW
    block_w = src_w // WINDOW
    inner_h = src_h - block_h * WINDOW
    inner_w = src_w - block_w * WINDOW
    window_row = (
        (((batch * BLOCKS_H + block_h) * BLOCKS_W + block_w) * WINDOW + inner_h)
        * WINDOW
        + inner_w
    )

    src_offsets = window_row * HIDDEN + cols
    dst_offsets = row_ids * HIDDEN + cols

    window_vals = tl.load(
        window_ptr + src_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual_vals = tl.load(
        residual_ptr + dst_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    add_bf16 = _f32_add(residual_vals, window_vals).to(tl.bfloat16)
    tl.store(add_out_ptr + dst_offsets, add_bf16, mask=mask)

    x = add_bf16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = (
        tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1)[:, None]
        / HIDDEN
    )
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

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
    normalized = _f32_mul(centered, invstd)
    y = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(norm_out_ptr + dst_offsets, y.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 8abb13ef: (T([401408,128], bf16), T([128,56,56,128], bf16), T([128], bf16), T([128], bf16), ...)
@oracle_impl(hardware="B200", point="8abb13ef", BLOCK_H=128, ROW_BLOCK=8, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2, _shape3, shape4, shape5 = inputs
    batch = int(arg1_1.shape[0])
    height = int(arg1_1.shape[1])
    width = int(arg1_1.shape[2])
    hidden = int(arg1_1.shape[3])
    tokens = height * width
    rows = batch * tokens
    window = 7
    blocks_h = height // window
    blocks_w = width // window
    add_shape = tuple(int(dim) if int(dim) > 0 else tokens for dim in shape4)
    norm_shape = tuple(int(dim) for dim in shape5)

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _swin_shifted_reverse_residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        ROWS=rows,
        TOKENS=tokens,
        HEIGHT=height,
        WIDTH=width,
        HIDDEN=hidden,
        WINDOW=window,
        BLOCKS_H=blocks_h,
        BLOCKS_W=blocks_w,
        SHIFT=53,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
