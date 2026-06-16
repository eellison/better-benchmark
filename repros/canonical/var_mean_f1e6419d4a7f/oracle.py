"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin window-reverse residual bf16 LayerNorm scope, including the window reverse view/permute/clone remap, bf16 residual add returned as `[128,196,512]`, explicit bf16-to-fp32 cast before population `var_mean`, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and `[25088,512]` view output in one Triton row kernel, whereas Inductor lowers the captured reshape/permute/clone/add/cast/var_mean/affine/cast/view graph through generic normalization and layout scheduling; Inductor cannot do this today because its normalization scheduler does not retain a nontrivial window-reverse producer and the returned pre-norm bf16 side output across the row-statistics pass and affine epilogue; the fix is SCHEDULER_FUSION: extend the fixed-hidden LayerNorm template to inline window-reverse producers and emit both the pre-norm bf16 row store and normalized bf16 row store from one row tile."""

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
def _swin_window_reverse_residual_layernorm_kernel(
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
    block_h = h // WINDOW
    block_w = w // WINDOW
    inner_h = h - block_h * WINDOW
    inner_w = w - block_w * WINDOW
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
    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1)[:, None] / HIDDEN
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
    y = _f32_add(scaled, bias)
    tl.store(norm_out_ptr + dst_offsets, y.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 2802cf0f: (T([25088,512], bf16), T([128,14,14,512], bf16), T([512], bf16), T([512], bf16), ...)
@oracle_impl(hardware="B200", point="2802cf0f", BLOCK_H=512, ROW_BLOCK=4, num_warps=4, num_stages=3)
# 536b6e86: (T([100352,256], bf16), T([128,28,28,256], bf16), T([256], bf16), T([256], bf16), ...)
@oracle_impl(hardware="B200", point="536b6e86", BLOCK_H=256, ROW_BLOCK=8, num_warps=4, num_stages=3)
# 8abb13ef: (T([401408,128], bf16), T([128,56,56,128], bf16), T([128], bf16), T([128], bf16), ...)
@oracle_impl(hardware="B200", point="8abb13ef", BLOCK_H=128, ROW_BLOCK=8, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2, _shape3, _shape4, _shape5 = inputs
    batch = int(arg1_1.shape[0])
    height = int(arg1_1.shape[1])
    width = int(arg1_1.shape[2])
    hidden = int(arg1_1.shape[3])
    tokens = height * width
    rows = batch * tokens
    window = 7
    blocks_h = height // window
    blocks_w = width // window
    add_shape = (batch, tokens, hidden)
    norm_shape = (rows, hidden)

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

    _swin_window_reverse_residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
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
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
