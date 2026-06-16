"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MobileViT residual-add LayerNorm and patch-fold scope, materializing the returned bf16 add, fp32 mean and rsqrt side outputs, and final bf16 NCHW fold while preserving the returned bf16 add and final-cast boundaries, whereas Inductor lowers the add, var_mean/rsqrt/affine, and two contiguous patch-layout rewrites as generic reduction and clone/permute kernels; Inductor cannot do this today because its norm-template canonicalization does not recognize this MobileViT residual LayerNorm with observable side outputs and a patch-fold epilogue as one full-scope template; the fix is NEW_PATTERN: add a MobileViT residual LayerNorm fold lowering that preserves the compiled bf16-to-fp32 normalization path, emits the side mean/rsqrt outputs, and sinks the affine plus patch-fold indexing into the output plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5
PATCH_LANES = 4
PATCH_H = 2
PATCH_W = 2


@triton.jit
def _layernorm_stats_kernel(
    addmm_ptr,
    residual_ptr,
    add_out_ptr,
    mean_ptr,
    rsqrt_ptr,
    total_rows: tl.constexpr,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
    row_block: tl.constexpr,
):
    rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
    cols = tl.arange(0, block_h)[None, :]
    row_mask = rows < total_rows
    col_mask = cols < hidden
    mask = row_mask & col_mask

    offsets = rows * hidden + cols
    values = (
        tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    tl.store(add_out_ptr + offsets, values, mask=mask)
    values_for_sum = tl.where(mask, values, 0.0)
    mean = tl.sum(values_for_sum, axis=1)[:, None] / hidden
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
    invstd = tl.rsqrt(variance + eps)

    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(rsqrt_ptr + rows, invstd, mask=row_mask)


@triton.jit
def _affine_patch_fold_kernel(
    add_out_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
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

    x = tl.load(add_out_ptr + row * hidden + c, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row, mask=mask, other=0.0)
    invstd = tl.load(rsqrt_ptr + row, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * invstd
    y = y * weight
    y = y + bias
    tl.store(output_ptr + offsets, y, mask=mask)


@oracle_impl(
    hardware="B200",
    point="f3a46541",
    HIDDEN=144,
    PATCHES=256,
    HEIGHT=32,
    WIDTH=32,
    ROW_BLOCK=4,
    BLOCK_H=256,
    FOLD_BLOCK=1024,
    STATS_WARPS=1,
    FOLD_WARPS=4,
)
@oracle_impl(
    hardware="B200",
    point="f6aa1a84",
    HIDDEN=192,
    PATCHES=64,
    HEIGHT=16,
    WIDTH=16,
    ROW_BLOCK=4,
    BLOCK_H=256,
    FOLD_BLOCK=1024,
    STATS_WARPS=1,
    FOLD_WARPS=4,
)
@oracle_impl(
    hardware="B200",
    point="74bd5ffe",
    HIDDEN=240,
    PATCHES=16,
    HEIGHT=8,
    WIDTH=8,
    ROW_BLOCK=4,
    BLOCK_H=256,
    FOLD_BLOCK=1024,
    STATS_WARPS=1,
    FOLD_WARPS=4,
)
def oracle_forward(
    inputs,
    *,
    HIDDEN: int,
    PATCHES: int,
    HEIGHT: int,
    WIDTH: int,
    ROW_BLOCK: int,
    BLOCK_H: int,
    FOLD_BLOCK: int,
    STATS_WARPS: int,
    FOLD_WARPS: int,
):
    addmm, residual, weight, bias, _shape0, _shape1, _shape2, _shape3 = inputs
    total_rows = addmm.numel() // HIDDEN
    groups = total_rows // PATCHES
    batch = groups // PATCH_LANES

    add_out = torch.empty_like(residual)
    mean = torch.empty((groups, PATCHES, 1), device=addmm.device, dtype=torch.float32)
    rsqrt = torch.empty((groups, PATCHES, 1), device=addmm.device, dtype=torch.float32)
    output = torch.empty((batch, HIDDEN, HEIGHT, WIDTH), device=addmm.device, dtype=torch.bfloat16)

    _layernorm_stats_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        addmm,
        residual,
        add_out,
        mean,
        rsqrt,
        total_rows,
        HIDDEN,
        EPS,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=STATS_WARPS,
        num_stages=3,
    )
    _affine_patch_fold_kernel[(triton.cdiv(output.numel(), FOLD_BLOCK),)](
        add_out,
        mean,
        rsqrt,
        weight,
        bias,
        output,
        output.numel(),
        HIDDEN,
        PATCHES,
        WIDTH // PATCH_W,
        HEIGHT,
        WIDTH,
        PATCH_LANES,
        PATCH_H,
        PATCH_W,
        FOLD_BLOCK,
        num_warps=FOLD_WARPS,
        num_stages=3,
    )
    return add_out, mean, rsqrt, output
