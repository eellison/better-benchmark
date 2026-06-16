"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 MT5/T5 residual layer-norm-backward return tuple by row-tiling the `[M, 512]` producer, sharing each row-local reduction with the f32 residual epilogue, preserving the bf16 dropout-mask rounding boundary for the flattened/transpose side output, and emitting partial column sums from the same element pass, whereas Inductor currently schedules the row reduction, dependent residual epilogue, bf16 materialization, transpose view, and sibling column reduction as separate generic work over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local reductions, explicit dtype-conversion boundaries, layout-changing side outputs, and compatible column accumulators in one producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer that fuses the full layer-norm-backward epilogue, writes the side outputs in target layout, emits column partials, and finalizes them together."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


D = 512
DROP_SCALE = 1.1111111111111112
INV_D = 1.0 / D
BLOCK_D = 512


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_tile_kernel(
    x_bf16_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    residual_ptr,
    keep_ptr,
    partial_col_ptr,
    add_out_ptr,
    bf16_out_ptr,
    M: tl.constexpr,
    D_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    INV_D_: tl.constexpr,
    ROW_TILE: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    tile_m = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D_)
    col_mask = cols < D_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    partial_col = tl.zeros((BLOCK_D_,), dtype=tl.float32)

    for row_i in tl.range(0, ROW_TILE):
        row = tile_m * ROW_TILE + row_i
        row_mask = row < M
        mask = row_mask & col_mask
        offsets = row * D_ + cols

        x = tl.load(x_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)

        weighted = _mul_rn(x, weight)
        rhs_scaled = _mul_rn(rhs, row_scale)
        partial_col += tl.where(mask, _mul_rn(x, rhs_scaled), 0.0)

        row_terms = _mul_rn(weighted, rhs)
        row_sum = tl.sum(tl.where(mask, row_terms, 0.0), axis=0)
        direct = _mul_rn(weighted, row_scale)
        add = _add_rn(residual, direct)

        row_scale_sq = _mul_rn(row_scale, row_scale)
        row_scale_cu = _mul_rn(row_scale_sq, row_scale)
        scaled_sum = _mul_rn(_mul_rn(row_sum, -0.5), row_scale_cu)
        div = _mul_rn(scaled_sum, INV_D_)
        correction = _mul_rn(div, _mul_rn(rhs, 2.0))
        add_out = _add_rn(add, correction)

        add_bf16 = add_out.to(tl.bfloat16)
        keep_scale = _mul_rn(keep, DROP_SCALE_).to(tl.bfloat16)
        bf16_out = _mul_rn(add_bf16.to(tl.float32), keep_scale.to(tl.float32)).to(
            tl.bfloat16
        )

        tl.store(add_out_ptr + offsets, add_out, mask=mask)
        tl.store(bf16_out_ptr + offsets, bf16_out, mask=mask)

    tl.store(partial_col_ptr + tile_m * D_ + cols, partial_col, mask=col_mask)


@triton.jit
def _finalize_column_sum_kernel(
    partial_col_ptr,
    out_col_ptr,
    NUM_ROW_TILES: tl.constexpr,
    D_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < D_)
    offsets = tiles[:, None] * D_ + cols[None, :]
    values = tl.load(partial_col_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_col_ptr + cols, tl.sum(values, axis=0), mask=cols < D_)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 1c935b51: (T([4096,512], bf16), T([512], f32), T([32,128,512], f32), ...)
# a4c9a272: (T([8192,512], bf16), T([512], f32), T([8,1024,512], f32), ...)
@oracle_impl(hardware="B200", point="1c935b51", ROW_TILE=4, FINAL_BLOCK_COLS=8, num_warps=4)
@oracle_impl(hardware="B200", point="a4c9a272", ROW_TILE=4, FINAL_BLOCK_COLS=8, num_warps=4)
def oracle_forward(
    inputs,
    *,
    ROW_TILE: int,
    FINAL_BLOCK_COLS: int,
    num_warps: int,
):
    (
        x_bf16,
        weight,
        rhs,
        row_scale,
        residual,
        keep,
        full_shape_param,
        sum_shape_param,
        _expanded_shape_param,
        flat_shape_param,
    ) = inputs

    m = int(x_bf16.shape[0])
    num_row_tiles = triton.cdiv(m, ROW_TILE)
    final_block_tiles = 1 << (num_row_tiles - 1).bit_length()

    add_out = torch.empty_strided(
        _shape_tuple(full_shape_param),
        (int(full_shape_param[1]) * D, D, 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        _shape_tuple(flat_shape_param),
        (D, 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    partial_col = torch.empty_strided(
        (num_row_tiles, D),
        (D, 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    out_col = torch.empty_strided(
        _shape_tuple(sum_shape_param),
        (1,),
        device=x_bf16.device,
        dtype=torch.float32,
    )

    _row_tile_kernel[(num_row_tiles,)](
        x_bf16,
        weight,
        rhs,
        row_scale,
        residual,
        keep,
        partial_col,
        add_out,
        bf16_out,
        M=m,
        D_=D,
        DROP_SCALE_=DROP_SCALE,
        INV_D_=INV_D,
        ROW_TILE=ROW_TILE,
        BLOCK_D_=BLOCK_D,
        num_warps=num_warps,
    )
    _finalize_column_sum_kernel[(triton.cdiv(D, FINAL_BLOCK_COLS),)](
        partial_col,
        out_col,
        NUM_ROW_TILES=num_row_tiles,
        D_=D,
        BLOCK_TILES=final_block_tiles,
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=8,
    )
    return out_col, add_out, bf16_out, bf16_out.permute(1, 0)
