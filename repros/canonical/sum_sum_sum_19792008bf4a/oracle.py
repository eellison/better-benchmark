"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BERT bf16 layer-norm-backward return tuple by row-tiling the `[2048, 768]` bf16 producer, preserving the visible f32 `add_3` tensor, the bf16 dropout-masked `[2048, 768]` output and its `[768, 2048]` permute alias, and the three compatible `[768]` column reductions from the same row-tile producer, whereas Inductor lowers the row-local hidden reductions, bf16/dropout cast epilogue, output aliasing, and sibling column reductions as generic fused regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, dtype-rounding epilogues, a dependent full-tensor side store, and multiple column accumulators tied to one producer schedule; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across token-row tiles, fuse the visible f32 and bf16 side stores, and finalize all sibling column partials together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


HIDDEN = 768
EPS = 1.0e-6
INV_HIDDEN = 1.0 / HIDDEN
ROW_BACKWARD_SCALE = 0.002607561929595828
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _row_stats_kernel(
    x_ptr,
    gamma_ptr,
    dy_ptr,
    denom_base_ptr,
    full_ptr,
    row_neg_x_gamma_sum_ptr,
    row_coef_ptr,
    ROWS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    EPS_: tl.constexpr,
    ROW_BACKWARD_SCALE_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row_block = tl.program_id(0)
    rows = row_block * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN_
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN_ + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    denom_base = tl.load(denom_base_ptr + rows, mask=row_mask, other=0.0).to(
        tl.float32
    )
    inv_denom = 1.0 / (denom_base + EPS_)

    x_gamma_over_denom = (x * inv_denom[:, None]) * gamma[None, :]
    row_curvature_sum = -tl.sum(
        tl.where(mask, x_gamma_over_denom * dy * inv_denom[:, None], 0.0),
        axis=1,
    )
    row_neg_x_gamma_sum = tl.sum(
        tl.where(mask, -x_gamma_over_denom, 0.0),
        axis=1,
    )
    full_value = tl.load(full_ptr).to(tl.float32)
    row_coef = tl.where(
        denom_base == 0.0,
        full_value,
        row_curvature_sum / (denom_base * 2.0),
    )
    row_coef = row_coef * ROW_BACKWARD_SCALE_

    tl.store(row_neg_x_gamma_sum_ptr + rows, row_neg_x_gamma_sum, mask=row_mask)
    tl.store(row_coef_ptr + rows, row_coef, mask=row_mask)


@triton.jit
def _materialize_and_partial_sum_kernel(
    x_ptr,
    gamma_ptr,
    dy_ptr,
    denom_base_ptr,
    residual_ptr,
    keep_ptr,
    row_neg_x_gamma_sum_ptr,
    row_coef_ptr,
    partial_sum_x_ptr,
    partial_sum_x_dy_over_denom_ptr,
    partial_sum_out_ptr,
    add_out_ptr,
    bf16_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    INV_HIDDEN_: tl.constexpr,
    EPS_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN_
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN_ + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    denom_base = tl.load(denom_base_ptr + rows, mask=row_mask, other=0.0).to(
        tl.float32
    )
    inv_denom = 1.0 / (denom_base + EPS_)

    x_over_denom = x * inv_denom[:, None]
    x_gamma_over_denom = x_over_denom * gamma[None, :]
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_coef = tl.load(row_coef_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_neg_x_gamma_sum = tl.load(
        row_neg_x_gamma_sum_ptr + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)

    add_1 = residual + x_gamma_over_denom
    add_2 = add_1 + row_coef[:, None] * dy
    add_3 = add_2 + row_neg_x_gamma_sum[:, None] * INV_HIDDEN_
    tl.store(add_out_ptr + offsets, add_3, mask=mask)

    keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
    keep_scaled = (keep.to(tl.bfloat16).to(tl.float32) * DROPOUT_SCALE_).to(
        tl.bfloat16
    )
    bf16_out = (add_3.to(tl.bfloat16).to(tl.float32) * keep_scaled.to(tl.float32)).to(
        tl.bfloat16
    )
    tl.store(bf16_out_ptr + offsets, bf16_out, mask=mask)

    partial_offsets = row_block * HIDDEN_ + cols
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_x_dy_over_denom = tl.sum(tl.where(mask, x_over_denom * dy, 0.0), axis=0)
    sum_out = tl.sum(tl.where(mask, bf16_out.to(tl.float32), 0.0), axis=0)
    tl.store(partial_sum_x_ptr + partial_offsets, sum_x, mask=col_mask)
    tl.store(
        partial_sum_x_dy_over_denom_ptr + partial_offsets,
        sum_x_dy_over_denom,
        mask=col_mask,
    )
    tl.store(partial_sum_out_ptr + partial_offsets, sum_out, mask=col_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_x_ptr,
    partial_sum_x_dy_over_denom_ptr,
    partial_sum_out_ptr,
    out_sum_x_ptr,
    out_sum_x_dy_over_denom_ptr,
    out_sum_out_ptr,
    NUM_ROW_TILES: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_sum_x = tl.zeros((BLOCK_C,), tl.float32)
    acc_sum_x_dy_over_denom = tl.zeros((BLOCK_C,), tl.float32)
    acc_sum_out = tl.zeros((BLOCK_C,), tl.float32)
    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_base + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & col_mask[None, :]
        offsets = tiles[:, None] * HIDDEN_ + cols[None, :]
        acc_sum_x += tl.sum(
            tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0),
            axis=0,
        )
        acc_sum_x_dy_over_denom += tl.sum(
            tl.load(
                partial_sum_x_dy_over_denom_ptr + offsets,
                mask=mask,
                other=0.0,
            ),
            axis=0,
        )
        acc_sum_out += tl.sum(
            tl.load(partial_sum_out_ptr + offsets, mask=mask, other=0.0),
            axis=0,
        )

    tl.store(out_sum_x_ptr + cols, acc_sum_x, mask=col_mask)
    tl.store(
        out_sum_x_dy_over_denom_ptr + cols,
        acc_sum_x_dy_over_denom,
        mask=col_mask,
    )
    tl.store(
        out_sum_out_ptr + cols,
        acc_sum_out.to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# e4a199c7: BERT train, rows=2048, hidden=768.
@oracle_impl(
    hardware="B200",
    point="e4a199c7",
    STATS_ROW_BLOCK=4,
    STATS_BLOCK_H=1024,
    ELEM_ROW_BLOCK=32,
    ELEM_BLOCK_C=128,
    FINAL_BLOCK_C=16,
    FINAL_BLOCK_TILES=64,
    STATS_WARPS=4,
    ELEM_WARPS=4,
    FINAL_WARPS=8,
)
def oracle_forward(
    inputs,
    *,
    STATS_ROW_BLOCK: int,
    STATS_BLOCK_H: int,
    ELEM_ROW_BLOCK: int,
    ELEM_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    FINAL_BLOCK_TILES: int,
    STATS_WARPS: int,
    ELEM_WARPS: int,
    FINAL_WARPS: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
    ) = inputs
    del _shape_param_0

    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    num_stats_tiles = triton.cdiv(rows, STATS_ROW_BLOCK)
    num_row_tiles = triton.cdiv(rows, ELEM_ROW_BLOCK)
    device = arg0_1.device

    row_neg_x_gamma_sum = torch.empty((rows,), device=device, dtype=torch.float32)
    row_coef = torch.empty((rows,), device=device, dtype=torch.float32)

    partial_sum_x = torch.empty(
        (num_row_tiles, hidden),
        device=device,
        dtype=torch.float32,
    )
    partial_sum_x_dy_over_denom = torch.empty(
        (num_row_tiles, hidden),
        device=device,
        dtype=torch.float32,
    )
    partial_sum_out = torch.empty(
        (num_row_tiles, hidden),
        device=device,
        dtype=torch.float32,
    )

    add_shape = tuple(int(dim) for dim in _shape_param_3)
    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=device,
        dtype=torch.float32,
    )
    out_shape = tuple(int(dim) for dim in _shape_param_4)
    bf16_out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=device,
        dtype=torch.bfloat16,
    )

    _row_stats_kernel[(num_stats_tiles,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg5_1,
        row_neg_x_gamma_sum,
        row_coef,
        ROWS=rows,
        HIDDEN_=hidden,
        EPS_=EPS,
        ROW_BACKWARD_SCALE_=ROW_BACKWARD_SCALE,
        ROW_BLOCK=STATS_ROW_BLOCK,
        BLOCK_H=STATS_BLOCK_H,
        num_warps=STATS_WARPS,
    )

    _materialize_and_partial_sum_kernel[
        (num_row_tiles, triton.cdiv(hidden, ELEM_BLOCK_C))
    ](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg6_1,
        row_neg_x_gamma_sum,
        row_coef,
        partial_sum_x,
        partial_sum_x_dy_over_denom,
        partial_sum_out,
        add_out,
        bf16_out,
        ROWS=rows,
        HIDDEN_=hidden,
        INV_HIDDEN_=INV_HIDDEN,
        EPS_=EPS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ROW_BLOCK=ELEM_ROW_BLOCK,
        BLOCK_C=ELEM_BLOCK_C,
        num_warps=ELEM_WARPS,
    )

    sum_x = torch.empty((hidden,), device=device, dtype=torch.float32)
    sum_x_dy_over_denom = torch.empty((hidden,), device=device, dtype=torch.float32)
    sum_out = torch.empty((hidden,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(hidden, FINAL_BLOCK_C),)](
        partial_sum_x,
        partial_sum_x_dy_over_denom,
        partial_sum_out,
        sum_x,
        sum_x_dy_over_denom,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        HIDDEN_=hidden,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=FINAL_WARPS,
    )

    view_3 = bf16_out.view(out_shape)
    return (
        sum_x.view(tuple(int(dim) for dim in _shape_param_1)),
        sum_x_dy_over_denom.view(tuple(int(dim) for dim in _shape_param_2)),
        add_out,
        view_3,
        view_3.permute(1, 0),
        sum_out.view(tuple(int(dim) for dim in _shape_param_5)),
    )
