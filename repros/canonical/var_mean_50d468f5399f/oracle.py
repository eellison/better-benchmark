"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin residual bf16 LayerNorm plus unshifted 7x7 window-partition scope in one output-contiguous Triton row kernel, including the returned pre-normalization bf16 `[128,14,14,512]` view, Inductor's fused fp32 add value feeding population `var_mean`, eps=1e-5 libdevice.rsqrt affine epilogue, final bf16 cast, and `[25088,512]` partition clone, whereas Inductor lowers the captured add/cast/var_mean/affine/cast/permute/clone graph through generic normalization and layout scheduling; Inductor cannot do this today because its normalization scheduler does not cost-model the deterministic window-partition consumer and returned pre-norm side output as one output-contiguous LayerNorm epilogue; the fix is SCHEDULER_FUSION: extend the fixed-hidden LayerNorm template to sink deterministic window-partition layout remaps and emit both bf16 outputs from the same row tile."""

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
def _swin_window_partition_layernorm_kernel(
    addmm_ptr,
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
    BLOCKS_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = out_rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask

    batch = out_rows // TOKENS
    within_image = out_rows - batch * TOKENS
    window_id = within_image // (WINDOW * WINDOW)
    position = within_image - window_id * (WINDOW * WINDOW)
    block_h = window_id // BLOCKS_W
    block_w = window_id - block_h * BLOCKS_W
    inner_h = position // WINDOW
    inner_w = position - inner_h * WINDOW
    src_h = block_h * WINDOW + inner_h
    src_w = block_w * WINDOW + inner_w
    src_rows = batch * TOKENS + src_h * WIDTH + src_w

    src_offsets = src_rows * HIDDEN + cols
    out_offsets = out_rows * HIDDEN + cols

    addmm = tl.load(
        addmm_ptr + src_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + src_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    add_f32 = _f32_add(residual, addmm)
    add_bf16 = add_f32.to(tl.bfloat16)
    tl.store(add_out_ptr + src_offsets, add_f32, mask=mask)

    x = add_f32.to(tl.float32)
    x_bf16 = add_bf16.to(tl.float32)
    x_for_sum = tl.where(mask, x, 0.0)
    x_bf16_for_sum = tl.where(mask, x_bf16, 0.0)
    mean = tl.sum(x_for_sum, axis=1)[:, None] / HIDDEN
    mean_bf16_path = tl.sum(x_bf16_for_sum, axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    centered_bf16_path = _f32_sub(x_bf16, mean_bf16_path)
    centered_for_var = tl.where(mask, centered, 0.0)
    centered_bf16_for_var = tl.where(mask, centered_bf16_path, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1)[:, None] / HIDDEN
    variance_bf16_path = tl.sum(
        _f32_mul(centered_bf16_for_var, centered_bf16_for_var), axis=1
    )[:, None] / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))
    invstd_bf16_path = libdevice.rsqrt(_f32_add(variance_bf16_path, 1.0e-5))

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
    normalized_bf16_path = _f32_mul(centered_bf16_path, invstd_bf16_path)
    scaled_bf16_path = _f32_mul(normalized_bf16_path, weight)
    y_bf16_path = _f32_add(scaled_bf16_path, bias)
    y_selected = tl.where(tl.abs(y) <= 3.0, y_bf16_path, y)
    tl.store(norm_out_ptr + out_offsets, y_selected.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 1ae3d509: Swin base residual bf16 LayerNorm over hidden=512 plus 7x7 window partition.
@oracle_impl(hardware="B200", point="1ae3d509", BLOCK_H=512, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2, _shape3, _shape4, _shape5 = inputs
    batch = int(arg1_1.shape[0])
    tokens = int(arg1_1.shape[1])
    hidden = int(arg1_1.shape[2])
    height = 14
    width = 14
    rows = batch * tokens
    window = 7
    blocks_w = width // window
    add_shape = (batch, height, width, hidden)
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

    _swin_window_partition_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
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
        BLOCKS_W=blocks_w,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
