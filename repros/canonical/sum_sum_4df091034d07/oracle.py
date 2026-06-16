"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MT5 bf16/f32 layer-norm-backward fragment by row-tiling the `[4096, 512]` producer, sharing the row-local reduction needed by the full f32 epilogue, emitting bf16 dropout-masked output plus its returned permuted alias, and accumulating the sibling `[512]` column reduction from the same element pass, whereas Inductor currently schedules the row reduction, full-size epilogue, bf16 cast/dropout product, layout view, and sibling column reduction as separate generic regions over replayed or materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local reductions, compatible column accumulators, dtype-cast boundaries, and required aliasing view outputs in one producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer that fuses these reductions and epilogues while preserving bf16 rounding and view alias semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 4096
D = 512
DROPOUT_SCALE = 1.1111111111111112
INV_D = 0.001953125


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mt5_row_tile_kernel(
    arg0_ptr,
    arg1_ptr,
    weight_ptr,
    norm_arg_ptr,
    row_scale_ptr,
    residual_ptr,
    mask_ptr,
    partial_col_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    M_: tl.constexpr,
    D_: tl.constexpr,
    ROW_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    INV_D_: tl.constexpr,
):
    tile_m = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    partial_col = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for row_i in tl.range(0, ROW_TILE):
        row = tile_m * ROW_TILE + row_i
        mask = (row < M_) & col_mask
        offsets = row * D_ + cols

        x0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add_value = _add_rn_f32(x0, x1)

        norm_arg = tl.load(norm_arg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row < M_, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        norm_scaled = _mul_rn_f32(norm_arg, row_scale)
        partial_col += tl.where(mask, _mul_rn_f32(add_value, norm_scaled), 0.0)

        weighted = _mul_rn_f32(add_value, weight)
        row_terms = _mul_rn_f32(weighted, norm_arg)
        row_sum = tl.sum(tl.where(mask, row_terms, 0.0), axis=0)

        direct = _mul_rn_f32(weighted, row_scale)
        add_1 = _add_rn_f32(residual, direct)

        scale_sq = _mul_rn_f32(row_scale, row_scale)
        scale_cu = _mul_rn_f32(scale_sq, row_scale)
        mul_5 = _mul_rn_f32(row_sum, -0.5)
        mul_6 = _mul_rn_f32(mul_5, scale_cu)
        div = _mul_rn_f32(mul_6, INV_D_)
        mul_7 = _mul_rn_f32(norm_arg, 2.0)
        mul_8 = _mul_rn_f32(div, mul_7)
        add_2 = _add_rn_f32(add_1, mul_8)

        tl.store(out_f32_ptr + offsets, add_2, mask=mask)

        mask_bf16 = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.bfloat16)
        mask_scaled = (mask_bf16 * DROPOUT_SCALE_).to(tl.bfloat16)
        out_bf16 = (add_2.to(tl.bfloat16) * mask_scaled).to(tl.bfloat16)
        tl.store(out_bf16_ptr + offsets, out_bf16, mask=mask)

    tl.store(partial_col_ptr + tile_m * D_ + cols, partial_col, mask=col_mask)


@triton.jit
def _finalize_column_sum_kernel(
    partial_col_ptr,
    out_col_ptr,
    D_: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < D_
    tiles = tl.arange(0, BLOCK_TILES)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for tile_start in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tile_ids = tile_start + tiles
        mask = (tile_ids[:, None] < NUM_ROW_TILES) & col_mask[None, :]
        offsets = tile_ids[:, None] * D_ + cols[None, :]
        values = tl.load(partial_col_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(values, axis=0)

    tl.store(out_col_ptr + cols, acc, mask=col_mask)


@oracle_impl(hardware="B200", point="c98bcb56", ROW_TILE=4, FINAL_BLOCK_D=8, num_warps=8)
def oracle_forward(inputs, *, ROW_TILE, FINAL_BLOCK_D, num_warps):
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
    ) = inputs

    out_col = torch.empty((D,), device=arg0_1.device, dtype=torch.float32)
    out_f32 = torch.empty((32, 128, D), device=arg0_1.device, dtype=torch.float32)
    out_bf16 = torch.empty((M, D), device=arg0_1.device, dtype=torch.bfloat16)

    num_row_tiles = triton.cdiv(M, ROW_TILE)
    partial_col = torch.empty((num_row_tiles, D), device=arg0_1.device, dtype=torch.float32)

    _mt5_row_tile_kernel[(num_row_tiles,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        partial_col,
        out_f32,
        out_bf16,
        M_=M,
        D_=D,
        ROW_TILE=ROW_TILE,
        BLOCK_D=D,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        INV_D_=INV_D,
        num_warps=num_warps,
    )

    _finalize_column_sum_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_col,
        out_col,
        D_=D,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
    )

    return out_col, out_f32, out_bf16, out_bf16.permute(1, 0)
