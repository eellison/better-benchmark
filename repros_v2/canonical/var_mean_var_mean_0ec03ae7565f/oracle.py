"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 Swin dual channel-LayerNorm region, including the NCHW-channels-last to NHWC view, first population var_mean and affine epilogue rounded to the returned bf16 tensor, second population var_mean over that rounded bf16 result, second affine epilogue rounded to bf16, and the 7x7 window-partition clone/view by scattering rows directly to the final `[401408,128]` layout, whereas Inductor lowers the decomposed graph as generic norm-template kernels plus separate dtype-conversion and layout materialization work; Inductor cannot do this today because its scheduler has no chained LayerNorm plus Swin window-partition template that keeps the first rounded normalization in registers while also emitting the returned first-output tensor and sinking the final clone layout into the second normalization store; the fix is NEW_PATTERN: add a guarded Swin dual-LayerNorm partition lowering, or a general dependent-normalization fusion, that preserves bf16 rounding boundaries and emits the final layout directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 128
HEIGHT = 56
WIDTH = 56
GRID_H = 8
GRID_W = 8
WINDOW_H = 7
WINDOW_W = 7
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
def _swin_dual_layernorm_partition_kernel(
    x_ptr,
    weight1_ptr,
    bias1_ptr,
    weight2_ptr,
    bias2_ptr,
    first_out_ptr,
    window_out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    ROWS_N: tl.constexpr,
    HEIGHT_N: tl.constexpr,
    WIDTH_N: tl.constexpr,
    CHANNELS_N: tl.constexpr,
    GRID_H_N: tl.constexpr,
    GRID_W_N: tl.constexpr,
    WINDOW_H_N: tl.constexpr,
    WINDOW_W_N: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    source_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_C)[None, :]
    row_mask = source_rows < ROWS_N
    col_mask = cols < CHANNELS_N
    mask = row_mask & col_mask

    spatial = source_rows % (HEIGHT_N * WIDTH_N)
    batch = source_rows // (HEIGHT_N * WIDTH_N)
    src_h = spatial // WIDTH_N
    src_w = spatial - src_h * WIDTH_N

    x_offsets = batch * x_s0 + cols * x_s1 + src_h * x_s2 + src_w * x_s3
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    mean1 = (tl.sum(tl.where(mask, x, 0.0), axis=1) * (1.0 / CHANNELS_N))[:, None]
    centered1 = _f32_sub(x, mean1)
    var1 = (
        tl.sum(tl.where(mask, _f32_mul(centered1, centered1), 0.0), axis=1) * (1.0 / CHANNELS_N)
    )[:, None]
    invstd1 = libdevice.rsqrt(_f32_add(var1, 1.0e-5))
    w1 = tl.load(weight1_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    b1 = tl.load(bias1_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    y1 = _f32_add(_f32_mul(_f32_mul(centered1, invstd1), w1), b1).to(tl.bfloat16)

    first_offsets = source_rows * CHANNELS_N + cols
    tl.store(first_out_ptr + first_offsets, y1, mask=mask)

    y1_f32 = y1.to(tl.float32)
    mean2 = (tl.sum(tl.where(mask, y1_f32, 0.0), axis=1) * (1.0 / CHANNELS_N))[:, None]
    centered2 = _f32_sub(y1_f32, mean2)
    var2 = (
        tl.sum(tl.where(mask, _f32_mul(centered2, centered2), 0.0), axis=1) * (1.0 / CHANNELS_N)
    )[:, None]
    invstd2 = libdevice.rsqrt(_f32_add(var2, 1.0e-5))
    w2 = tl.load(weight2_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    b2 = tl.load(bias2_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    y2 = _f32_add(_f32_mul(_f32_mul(centered2, invstd2), w2), b2).to(tl.bfloat16)

    window_row = src_h // WINDOW_H_N
    inner_h = src_h - window_row * WINDOW_H_N
    window_col = src_w // WINDOW_W_N
    inner_w = src_w - window_col * WINDOW_W_N
    out_row = (
        (((batch * GRID_H_N + window_row) * GRID_W_N + window_col) * WINDOW_H_N + inner_h)
        * WINDOW_W_N
        + inner_w
    )
    tl.store(window_out_ptr + out_row * CHANNELS_N + cols, y2, mask=mask)


# b68c1040: Swin dual LayerNorm over hidden=128 with returned first bf16 NHWC tensor and final 7x7 window partition.
@oracle_impl(hardware="B200", point="b68c1040", ROW_BLOCK=32, BLOCK_C=128, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_C: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, _shape2, _shape3 = inputs
    first_out = torch.empty_strided(
        (BATCH, HEIGHT, WIDTH, CHANNELS),
        (HEIGHT * WIDTH * CHANNELS, WIDTH * CHANNELS, CHANNELS, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    window_out = torch.empty_strided(
        (ROWS, CHANNELS),
        (CHANNELS, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _swin_dual_layernorm_partition_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        first_out,
        window_out,
        arg0_1.stride(0),
        arg0_1.stride(1),
        arg0_1.stride(2),
        arg0_1.stride(3),
        ROWS_N=ROWS,
        HEIGHT_N=HEIGHT,
        WIDTH_N=WIDTH,
        CHANNELS_N=CHANNELS,
        GRID_H_N=GRID_H,
        GRID_W_N=GRID_W,
        WINDOW_H_N=WINDOW_H,
        WINDOW_W_N=WINDOW_W,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return first_out, window_out
