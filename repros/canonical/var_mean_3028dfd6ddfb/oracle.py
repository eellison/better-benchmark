"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT-S 2x2 patch-layout canonicalization and affine LayerNorm training scope in one Triton row kernel for all three captured channel/spatial points, including the channels-last `[128,C,H,W]` input clone-to-contiguous semantics, the view/permute/clone/view/permute/clone/view chain into the live bf16 `[512,patches,C]` output, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt and returned mean/rsqrt side outputs, fp32 weight/bias affine epilogue, and final bf16 `[512*patches,C]` view, whereas Inductor lowers the patch-layout materialization, normalization reduction, affine cast, and side-output stores through generic layout-copy and norm-template schedules; Inductor cannot do this today because its scheduler does not recognize this MobileViT patch-unfold producer as a direct row source for the fixed-hidden LayerNorm while preserving the visible pre-normalization layout tensor and saved statistic outputs; the fix is NEW_PATTERN: add a guarded MobileViT patch-unfold LayerNorm lowering that maps rows to original NCHW logical coordinates and fuses layout materialization, row statistics, affine cast, and sibling outputs into one full-scope plan."""

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
def _mobilevit_patch_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    layout_ptr,
    mean_ptr,
    rsqrt_ptr,
    out_ptr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    WIDTH: tl.constexpr,
    PATCHES: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask

    patch = rows % PATCHES
    row_group = rows // PATCHES
    lane = row_group % 4
    batch = row_group // 4
    half_width = WIDTH // 2
    patch_h = patch // half_width
    patch_w = patch - patch_h * half_width
    h = patch_h * 2 + lane // 2
    w = patch_w * 2 + lane - (lane // 2) * 2

    input_offsets = batch * stride_n + cols * stride_c + h * stride_h + w * stride_w
    output_offsets = rows * HIDDEN + cols

    x_bf16 = tl.load(
        x_ptr + input_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    values = x_bf16.to(tl.float32)
    tl.store(layout_ptr + output_offsets, x_bf16, mask=mask)

    sum_values = tl.sum(tl.where(mask, values, 0.0), axis=1)[:, None]
    mean = sum_values / HIDDEN
    centered = _f32_sub(values, mean)
    centered_masked = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1)[:, None] / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS))
    normalized = _f32_mul(centered, invstd)

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

    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(rsqrt_ptr + rows, invstd, mask=row_mask)
    tl.store(out_ptr + output_offsets, affine.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="11d32ae7", BLOCK_H=256, ROW_BLOCK=4, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="354f721b", BLOCK_H=256, ROW_BLOCK=4, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="3fce1def", BLOCK_H=256, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    layout_shape = _as_shape(_shape_param_2)
    out_shape = _as_shape(_shape_param_3)
    hidden = int(layout_shape[2])
    patches = int(layout_shape[1])
    rows = int(out_shape[0])
    stat_shape = (int(layout_shape[0]), patches, 1)

    layout = torch.empty_strided(
        layout_shape,
        _contiguous_stride(layout_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _mobilevit_patch_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        layout,
        mean,
        rsqrt,
        out,
        stride_n=arg0_1.stride(0),
        stride_c=arg0_1.stride(1),
        stride_h=arg0_1.stride(2),
        stride_w=arg0_1.stride(3),
        ROWS=rows,
        HIDDEN=hidden,
        WIDTH=int(arg0_1.shape[3]),
        PATCHES=patches,
        EPS=1.0e-5,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return layout, mean, rsqrt, out
