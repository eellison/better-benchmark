"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileViT bf16 layer-norm-backward layout tail by row-tiling the `[B*M,C]` source, sharing each hidden-row reduction, preserving the captured hardcoded `144` scale/divide and bf16 cast boundary before the residual add, writing the dependent residual result directly into the returned contiguous `[128,C,H,W]` 2x2 patch-fold layout, and cooperatively finalizing the two returned source column reductions from common partials, whereas Inductor schedules the row reductions, sibling column reductions, dependent bf16 pointwise tensor, and view/permute/clone patch-fold pipeline as separate generic regions over materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local layer-norm scalars, explicit bf16 rounding, required layout-changing side output, and compatible column partials in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add a MobileViT row-tiled reduction template that preserves literal constants and dtype boundaries while emitting the final patch-fold layout and all column reductions together."""

import torch
import triton
import triton.language as tl

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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mobilevit_layout_partials_kernel(
    x_ptr,
    weight_ptr,
    rhs_bf16_ptr,
    mean_ptr,
    inv_ptr,
    residual_bf16_ptr,
    add_out_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    PATCHES: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < CHANNELS
    offsets = rows[:, None] * CHANNELS + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    rhs_src = tl.load(rhs_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    weighted = _f32_mul(x, weight[None, :])
    rhs = _f32_mul(_f32_sub(rhs_src, mean[:, None]), inv[:, None])
    weighted_rhs = _f32_mul(weighted, rhs)
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)

    mul_1 = _f32_mul(weighted, 144.0)
    mul_4 = _f32_mul(rhs, row_dot[:, None])
    sub_1 = _f32_sub(mul_1, row_sum[:, None])
    sub_2 = _f32_sub(sub_1, mul_4)
    div = _f32_div(inv, 144.0)
    delta = _f32_mul(div[:, None], sub_2)
    delta_bf16 = delta.to(tl.bfloat16)
    add_bf16 = _f32_add(residual, delta_bf16.to(tl.float32)).to(tl.bfloat16)

    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    partial_cols = tl.arange(0, BLOCK_C)
    partial_mask = partial_cols < CHANNELS
    partial_offsets = tile * CHANNELS + partial_cols
    tl.store(
        partial_x_rhs_ptr + partial_offsets,
        tl.sum(tl.where(mask, _f32_mul(x, rhs), 0.0), axis=0),
        mask=partial_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=partial_mask,
    )


@triton.jit
def _mobilevit_layout_kernel(
    x_ptr,
    weight_ptr,
    rhs_bf16_ptr,
    mean_ptr,
    inv_ptr,
    residual_bf16_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    PATCHES: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < CHANNELS
    offsets = rows[:, None] * CHANNELS + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    rhs_src = tl.load(rhs_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    weighted = _f32_mul(x, weight[None, :])
    rhs = _f32_mul(_f32_sub(rhs_src, mean[:, None]), inv[:, None])
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, _f32_mul(weighted, rhs), 0.0), axis=1)

    sub_1 = _f32_sub(_f32_mul(weighted, 144.0), row_sum[:, None])
    sub_2 = _f32_sub(sub_1, _f32_mul(rhs, row_dot[:, None]))
    delta = _f32_mul(_f32_div(inv, 144.0)[:, None], sub_2)
    add_bf16 = _f32_add(residual, delta.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    group = rows // PATCHES
    patch = rows - group * PATCHES
    lane = group - (group // 4) * 4
    batch = group // 4
    patch_w = OUT_W // 2
    out_h = (patch // patch_w) * 2 + lane // 2
    out_w = (patch - (patch // patch_w) * patch_w) * 2 + lane - (lane // 2) * 2
    out_offsets = (
        (batch[:, None] * CHANNELS + cols[None, :]) * OUT_H + out_h[:, None]
    ) * OUT_W + out_w[:, None]
    tl.store(out_ptr + out_offsets, add_bf16, mask=mask)


@triton.jit
def _mobilevit_add_kernel(
    x_ptr,
    weight_ptr,
    rhs_bf16_ptr,
    mean_ptr,
    inv_ptr,
    residual_bf16_ptr,
    add_out_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < CHANNELS
    offsets = rows[:, None] * CHANNELS + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    rhs_src = tl.load(rhs_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    weighted = _f32_mul(x, weight[None, :])
    rhs = _f32_mul(_f32_sub(rhs_src, mean[:, None]), inv[:, None])
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, _f32_mul(weighted, rhs), 0.0), axis=1)

    sub_1 = _f32_sub(_f32_mul(weighted, 144.0), row_sum[:, None])
    sub_2 = _f32_sub(sub_1, _f32_mul(rhs, row_dot[:, None]))
    delta = _f32_mul(_f32_div(inv, 144.0)[:, None], sub_2)
    add_bf16 = _f32_add(residual, delta.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)


@triton.jit
def _column_partials_kernel(
    x_ptr,
    rhs_bf16_ptr,
    mean_ptr,
    inv_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < CHANNELS
    offsets = rows[:, None] * CHANNELS + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs_src = tl.load(rhs_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    rhs = _f32_mul(_f32_sub(rhs_src, mean[:, None]), inv[:, None])

    partial_offsets = row_block * CHANNELS + cols
    tl.store(
        partial_x_rhs_ptr + partial_offsets,
        tl.sum(tl.where(mask, _f32_mul(x, rhs), 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=col_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < CHANNELS
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * CHANNELS + cols[None, :]

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


@triton.jit
def _fold_row_major_to_nchw_tiled_kernel(
    add_out_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    PATCHES: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < CHANNELS
    mask = row_mask[:, None] & col_mask[None, :]

    group = rows // PATCHES
    patch = rows - group * PATCHES
    lane = group - (group // 4) * 4
    batch = group // 4
    patch_w = OUT_W // 2
    out_h = (patch // patch_w) * 2 + lane // 2
    out_w = (patch - (patch // patch_w) * patch_w) * 2 + lane - (lane // 2) * 2

    values = tl.load(
        add_out_ptr + rows[:, None] * CHANNELS + cols[None, :],
        mask=mask,
        other=0.0,
    )
    out_offsets = (
        (batch[:, None] * CHANNELS + cols[None, :]) * OUT_H + out_h[:, None]
    ) * OUT_W + out_w[:, None]
    tl.store(out_ptr + out_offsets, values, mask=mask)


@oracle_impl(
    hardware="B200",
    point="30b03cad",
    ROWS=131072,
    CHANNELS=144,
    PATCHES=256,
    OUT_H=32,
    OUT_W=32,
    BLOCK_M=32,
    BLOCK_C=256,
    FINAL_BLOCK_C=8,
    LAYOUT_ROW_BLOCK=256,
    LAYOUT_C_BLOCK=4,
    SPLIT_REDUCE=True,
    REDUCE_BLOCK_R=256,
    REDUCE_BLOCK_C=16,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="1c6da2dd",
    ROWS=32768,
    CHANNELS=192,
    PATCHES=64,
    OUT_H=16,
    OUT_W=16,
    BLOCK_M=64,
    BLOCK_C=256,
    FINAL_BLOCK_C=8,
    LAYOUT_ROW_BLOCK=256,
    LAYOUT_C_BLOCK=4,
    SPLIT_REDUCE=False,
    REDUCE_BLOCK_R=256,
    REDUCE_BLOCK_C=16,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="0c9dc299",
    ROWS=8192,
    CHANNELS=240,
    PATCHES=16,
    OUT_H=8,
    OUT_W=8,
    BLOCK_M=64,
    BLOCK_C=256,
    FINAL_BLOCK_C=8,
    LAYOUT_ROW_BLOCK=256,
    LAYOUT_C_BLOCK=4,
    SPLIT_REDUCE=False,
    REDUCE_BLOCK_R=256,
    REDUCE_BLOCK_C=16,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS: int,
    CHANNELS: int,
    PATCHES: int,
    OUT_H: int,
    OUT_W: int,
    BLOCK_M: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    LAYOUT_ROW_BLOCK: int,
    LAYOUT_C_BLOCK: int,
    SPLIT_REDUCE: bool,
    REDUCE_BLOCK_R: int,
    REDUCE_BLOCK_C: int,
    num_warps: int,
):
    x_bf16, weight, rhs_bf16, mean, inv, residual_bf16, *_shape_params = inputs
    add_groups = triton.cdiv(ROWS, BLOCK_M)
    reduce_groups = triton.cdiv(ROWS, REDUCE_BLOCK_R if SPLIT_REDUCE else BLOCK_M)
    partial_x_rhs = torch.empty(
        (reduce_groups, CHANNELS),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    partial_x = torch.empty(
        (reduce_groups, CHANNELS),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    out_x_rhs = torch.empty_strided(
        (CHANNELS,),
        (1,),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    out_x = torch.empty_strided(
        (CHANNELS,),
        (1,),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (128, CHANNELS, OUT_H, OUT_W),
        (CHANNELS * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )

    if SPLIT_REDUCE:
        _mobilevit_layout_kernel[(add_groups,)](
            x_bf16,
            weight,
            rhs_bf16,
            mean,
            inv,
            residual_bf16,
            out,
            ROWS=ROWS,
            CHANNELS=CHANNELS,
            PATCHES=PATCHES,
            OUT_H=OUT_H,
            OUT_W=OUT_W,
            BLOCK_M=BLOCK_M,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
        )
        _column_partials_kernel[
            (reduce_groups, triton.cdiv(CHANNELS, REDUCE_BLOCK_C))
        ](
            x_bf16,
            rhs_bf16,
            mean,
            inv,
            partial_x_rhs,
            partial_x,
            ROWS=ROWS,
            CHANNELS=CHANNELS,
            BLOCK_R=REDUCE_BLOCK_R,
            BLOCK_C=REDUCE_BLOCK_C,
            num_warps=8,
        )
    else:
        add_out = torch.empty_strided(
            (ROWS, CHANNELS),
            (CHANNELS, 1),
            device=x_bf16.device,
            dtype=torch.bfloat16,
        )
        _mobilevit_layout_partials_kernel[(add_groups,)](
            x_bf16,
            weight,
            rhs_bf16,
            mean,
            inv,
            residual_bf16,
            add_out,
            partial_x_rhs,
            partial_x,
            ROWS=ROWS,
            CHANNELS=CHANNELS,
            PATCHES=PATCHES,
            OUT_H=OUT_H,
            OUT_W=OUT_W,
            BLOCK_M=BLOCK_M,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
        )
        _fold_row_major_to_nchw_tiled_kernel[
            (triton.cdiv(ROWS, LAYOUT_ROW_BLOCK), triton.cdiv(CHANNELS, LAYOUT_C_BLOCK))
        ](
            add_out,
            out,
            ROWS=ROWS,
            CHANNELS=CHANNELS,
            PATCHES=PATCHES,
            OUT_H=OUT_H,
            OUT_W=OUT_W,
            BLOCK_R=LAYOUT_ROW_BLOCK,
            BLOCK_C=LAYOUT_C_BLOCK,
            num_warps=4,
        )
    group_block = triton.next_power_of_2(reduce_groups)
    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        out_x_rhs,
        out_x,
        NUM_GROUPS=reduce_groups,
        CHANNELS=CHANNELS,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )
    return out_x_rhs, out_x, out
