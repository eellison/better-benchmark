"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Swin residual-add LayerNorm and shifted-window partition scope in one Triton row kernel, including the returned bf16 `[128,56,56,128]` residual view, explicit bf16-to-fp32 cast before population `var_mean(..., correction=0)`, eps=1e-5 rsqrt affine epilogue, final bf16 cast, cyclic shift by three on both spatial axes, and direct 7x7 window-flattened `[401408,128]` store, whereas Inductor lowers the captured add/cast/var_mean/affine/cast/index/permute/clone/view graph through generic normalization plus separate layout/indexing schedules; Inductor cannot do this today because the normalization scheduler does not preserve a live returned bf16 producer while sinking deterministic cyclic-shift and window-partition indexing into the row-statistics store plan; the fix is SCHEDULER_FUSION: extend the fixed-hidden LayerNorm template to emit returned pre-norm stores and shifted window-partition output layouts from one full-scope row schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 56
WIDTH = 56
HIDDEN = 128
WINDOW = 7
SHIFT = 3
ROWS = BATCH * HEIGHT * WIDTH


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
def _swin_shifted_window_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS_N: tl.constexpr,
    HEIGHT_N: tl.constexpr,
    WIDTH_N: tl.constexpr,
    HIDDEN_N: tl.constexpr,
    WINDOW_N: tl.constexpr,
    SHIFT_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = out_rows < ROWS_N
    col_mask = cols < HIDDEN_N
    mask = row_mask & col_mask

    inner_w = out_rows % WINDOW_N
    tmp = out_rows // WINDOW_N
    inner_h = tmp % WINDOW_N
    tmp = tmp // WINDOW_N
    window_col = tmp % (WIDTH_N // WINDOW_N)
    tmp = tmp // (WIDTH_N // WINDOW_N)
    window_row = tmp % (HEIGHT_N // WINDOW_N)
    batch = tmp // (HEIGHT_N // WINDOW_N)

    shifted_h = window_row * WINDOW_N + inner_h
    shifted_w = window_col * WINDOW_N + inner_w
    src_h = (shifted_h + SHIFT_N) % HEIGHT_N
    src_w = (shifted_w + SHIFT_N) % WIDTH_N
    src_spatial = src_h * WIDTH_N + src_w
    src_row = batch * (HEIGHT_N * WIDTH_N) + src_spatial

    src_offsets = src_row * HIDDEN_N + cols
    flat_vals = tl.load(
        flat_ptr + src_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual_vals = tl.load(
        residual_ptr + src_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    add_bf16 = _f32_add(residual_vals, flat_vals).to(tl.bfloat16)
    tl.store(add_out_ptr + src_offsets, add_bf16, mask=mask)

    x = add_bf16.to(tl.float32)
    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=1)[:, None] / HIDDEN_N
    centered = _f32_sub(x, mean)
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1)[:, None] / HIDDEN_N
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
    scaled = _f32_mul(normalized, weight)
    out = _f32_add(scaled, bias).to(tl.bfloat16)

    dst_offsets = out_rows * HIDDEN_N + cols
    tl.store(norm_out_ptr + dst_offsets, out, mask=mask)


# 2038fe94: (T([401408,128], bf16), T([128,3136,128], bf16), T([128], bf16), T([128], bf16), ...)
@oracle_impl(hardware="B200", point="2038fe94", BLOCK_H=128, ROW_BLOCK=8, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2, _shape3, _shape4, _shape5 = inputs
    add_out = torch.empty_strided(
        (BATCH, HEIGHT, WIDTH, HIDDEN),
        (HEIGHT * WIDTH * HIDDEN, WIDTH * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _swin_shifted_window_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        ROWS_N=ROWS,
        HEIGHT_N=HEIGHT,
        WIDTH_N=WIDTH,
        HIDDEN_N=HIDDEN,
        WINDOW_N=WINDOW,
        SHIFT_N=SHIFT,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
