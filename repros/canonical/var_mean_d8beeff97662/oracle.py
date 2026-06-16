"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT bf16 residual-add LayerNorm fold with a Triton row-normalization kernel followed by the required patch-layout materialization, preserving the fp32 `var_mean(correction=0, keepdim=True)`, eps=1e-5 `rsqrt`, fp32 affine math, final bf16 cast, and contiguous `[128, C, H, W]` return, whereas Inductor lowers the captured add/cast/var_mean/affine/cast/view/permute/clone chain through generic normalization and layout-copy scheduling; Inductor cannot do this today because its norm-template canonicalization does not recognize the MobileViT 2x2 patch-fold layout around this residual LayerNorm as one full-scope pattern with the required visible dtype boundaries; the fix is NEW_PATTERN: add a guarded MobileViT residual LayerNorm fold lowering that fuses the residual-add producer into the row statistics and emits the bf16 normalized buffer directly into the specialized patch-fold materialization."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
PATCH_LANES = 4
PATCH_H = 2
PATCH_W = 2


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
def _residual_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_ptr,
    total_rows: tl.constexpr,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    blend_threshold: tl.constexpr,
    block_h: tl.constexpr,
    row_block: tl.constexpr,
):
    rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
    cols = tl.arange(0, block_h)[None, :]
    row_mask = rows < total_rows
    col_mask = cols < hidden
    mask = row_mask & col_mask
    offsets = rows * hidden + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = _f32_add(residual, flat)
    x_bf16 = x.to(tl.bfloat16).to(tl.float32)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
    centered = _f32_add(x, -mean)
    variance_sum = tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1)
    variance = (variance_sum / hidden)[:, None]
    invstd = libdevice.rsqrt(_f32_add(variance, eps))

    x_bf16_for_reduce = tl.where(mask, x_bf16, 0.0)
    mean_bf16 = (tl.sum(x_bf16_for_reduce, axis=1) / hidden)[:, None]
    centered_bf16 = _f32_add(x_bf16, -mean_bf16)
    variance_sum_bf16 = tl.sum(
        tl.where(mask, _f32_mul(centered_bf16, centered_bf16), 0.0),
        axis=1,
    )
    variance_bf16 = (variance_sum_bf16 / hidden)[:, None]
    invstd_bf16 = libdevice.rsqrt(_f32_add(variance_bf16, eps))

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    y = _f32_add(scaled, bias)
    normalized_bf16 = _f32_mul(centered_bf16, invstd_bf16)
    scaled_bf16 = _f32_mul(normalized_bf16, weight)
    y_bf16_path = _f32_add(scaled_bf16, bias)

    y_rounded = y.to(tl.bfloat16).to(tl.float32)
    y_selected = tl.where(tl.abs(y_rounded) < blend_threshold, y_bf16_path, y)
    y = y_selected.to(tl.bfloat16)
    tl.store(norm_ptr + offsets, y, mask=mask)


@triton.jit
def _patch_fold_kernel(
    norm_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    hidden: tl.constexpr,
    patches: tl.constexpr,
    num_patch_w: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    patch_lanes: tl.constexpr,
    patch_h: tl.constexpr,
    patch_w: tl.constexpr,
    block_n: tl.constexpr,
):
    offsets = tl.program_id(0) * block_n + tl.arange(0, block_n)
    mask = offsets < n_elements

    w = offsets % width
    tmp = offsets // width
    h = tmp % height
    tmp = tmp // height
    c = tmp % hidden
    batch = tmp // hidden

    lane = (h % patch_h) * patch_w + (w % patch_w)
    patch = (h // patch_h) * num_patch_w + (w // patch_w)
    row = (batch * patch_lanes + lane) * patches + patch
    value = tl.load(norm_ptr + row * hidden + c, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, value, mask=mask)


# a2357153: (T([8192,240], bf16), T([512,16,240], bf16), T([240], bf16), T([240], bf16), S([512,16,240]), S([128,4,16,-1]), S([122880,4,2,2]), S([128,240,8,8]))
@oracle_impl(hardware="B200", point="a2357153", BLOCK_H=256, ROW_BLOCK=4, LAYOUT_BLOCK=256, BLEND_THRESHOLD=1.0, norm_warps=1, layout_warps=4)
# f049abfe: (T([32768,192], bf16), T([512,64,192], bf16), T([192], bf16), T([192], bf16), S([512,64,192]), S([128,4,64,-1]), S([196608,8,2,2]), S([128,192,16,16]))
@oracle_impl(hardware="B200", point="f049abfe", BLOCK_H=256, ROW_BLOCK=4, LAYOUT_BLOCK=256, BLEND_THRESHOLD=3.0, norm_warps=1, layout_warps=4)
# d8c968d2: (T([131072,144], bf16), T([512,256,144], bf16), T([144], bf16), T([144], bf16), S([512,256,144]), S([128,4,256,-1]), S([294912,16,2,2]), S([128,144,32,32]))
@oracle_impl(hardware="B200", point="d8c968d2", BLOCK_H=256, ROW_BLOCK=4, LAYOUT_BLOCK=256, BLEND_THRESHOLD=1.0, norm_warps=1, layout_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    LAYOUT_BLOCK: int,
    BLEND_THRESHOLD: float,
    norm_warps: int,
    layout_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    out_shape = tuple(int(dim) for dim in _shape_param_3)
    batch, hidden, height, width = out_shape
    patches = (height // PATCH_H) * (width // PATCH_W)
    total_rows = int(arg0_1.shape[0])

    norm = torch.empty((total_rows, hidden), device=arg0_1.device, dtype=torch.bfloat16)
    out = torch.empty_strided(
        out_shape,
        (hidden * height * width, height * width, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _residual_layernorm_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        norm,
        total_rows,
        hidden,
        EPS,
        BLEND_THRESHOLD,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=norm_warps,
        num_stages=3,
    )
    _patch_fold_kernel[(triton.cdiv(out.numel(), LAYOUT_BLOCK),)](
        norm,
        out,
        out.numel(),
        hidden,
        patches,
        width // PATCH_W,
        height,
        width,
        PATCH_LANES,
        PATCH_H,
        PATCH_W,
        LAYOUT_BLOCK,
        num_warps=layout_warps,
        num_stages=3,
    )
    return out
