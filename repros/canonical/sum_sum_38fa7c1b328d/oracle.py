"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MT5 bf16/f32 layer-norm-backward fragment by row-tiling the `[4096, 512]` producer, preserving the sequential f32 residual-add chain across the sixteen bf16 residual inputs, sharing the row-local reduction needed by the full f32 epilogue, emitting the bf16 dropout-masked output plus its returned permuted alias, and accumulating the sibling `[512]` column reduction from the same element pass, whereas Inductor currently schedules the long residual chain, row reduction, full-size epilogue, bf16 cast/dropout product, layout view, and sibling column reduction as separate generic regions over replayed or materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates long f32 producer chains, row-local reductions, compatible column accumulators, dtype-cast boundaries, and required aliasing view outputs in one producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer that fuses these reductions and epilogues while preserving bf16 rounding and view alias semantics."""

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
def _mt5_residual_row_tile_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    arg10_ptr,
    arg11_ptr,
    arg12_ptr,
    arg13_ptr,
    arg14_ptr,
    arg15_ptr,
    arg16_ptr,
    mask1_ptr,
    weight_ptr,
    norm_arg_ptr,
    row_scale_ptr,
    mask2_ptr,
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
    mask_scale_bf16 = tl.full((BLOCK_D,), DROPOUT_SCALE_, tl.float32).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    for row_i in tl.range(0, ROW_TILE):
        row = tile_m * ROW_TILE + row_i
        mask = (row < M_) & col_mask
        offsets = row * D_ + cols

        residual = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = _add_rn_f32(
            residual, tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg11_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg12_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg13_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg14_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg15_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        residual = _add_rn_f32(
            residual, tl.load(arg16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )

        mask1 = tl.load(mask1_ptr + offsets, mask=mask, other=0).to(tl.float32)
        norm_arg = tl.load(norm_arg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row < M_, other=0.0).to(tl.float32)

        dropout_scale = _mul_rn_f32(mask1, DROPOUT_SCALE_)
        masked_residual = _mul_rn_f32(residual, dropout_scale)
        weighted = _mul_rn_f32(masked_residual, weight)
        norm_scaled = _mul_rn_f32(norm_arg, row_scale)
        partial_col = _add_rn_f32(
            partial_col,
            tl.where(mask, _mul_rn_f32(masked_residual, norm_scaled), 0.0),
        )

        row_terms = _mul_rn_f32(weighted, norm_arg)
        row_sum = tl.sum(tl.where(mask, row_terms, 0.0), axis=0)

        direct = _mul_rn_f32(weighted, row_scale)
        row_scale_sq = _mul_rn_f32(row_scale, row_scale)
        row_scale_cu = _mul_rn_f32(row_scale_sq, row_scale)
        correction_scale = _mul_rn_f32(_mul_rn_f32(row_sum, -0.5), row_scale_cu)
        correction_scale = _mul_rn_f32(correction_scale, INV_D_)
        correction = _mul_rn_f32(correction_scale, _mul_rn_f32(norm_arg, 2.0))
        add_value = _add_rn_f32(direct, correction)

        tl.store(out_f32_ptr + offsets, add_value, mask=mask)

        add_bf16 = add_value.to(tl.bfloat16, fp_downcast_rounding="rtne")
        mask2 = tl.load(mask2_ptr + offsets, mask=mask, other=0)
        scaled_mask = tl.where(
            mask2,
            mask_scale_bf16,
            tl.full((BLOCK_D,), 0.0, tl.bfloat16),
        )
        out_bf16 = _mul_rn_f32(
            add_bf16.to(tl.float32),
            scaled_mask.to(tl.float32),
        ).to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(out_bf16_ptr + offsets, out_bf16, mask=mask)

    tl.store(partial_col_ptr + tile_m * D_ + cols, partial_col, mask=col_mask)


@triton.jit
def _finalize_column_sum_kernel(
    partial_col_ptr,
    out_col_ptr,
    NUM_ROW_TILES: tl.constexpr,
    D_: tl.constexpr,
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


@oracle_impl(
    hardware="B200",
    point="c4d61901",
    ROW_TILE=4,
    BLOCK_D=512,
    FINAL_BLOCK_D=8,
    FINAL_BLOCK_TILES=1024,
)
def oracle_forward(
    inputs,
    *,
    ROW_TILE: int,
    BLOCK_D: int,
    FINAL_BLOCK_D: int,
    FINAL_BLOCK_TILES: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
        _shape_param_13,
        _shape_param_14,
        _shape_param_15,
        _shape_param_16,
        _shape_param_17,
        _shape_param_18,
    ) = inputs

    num_row_tiles = triton.cdiv(M, ROW_TILE)
    partial_col = torch.empty(
        (num_row_tiles, D), device=arg1_1.device, dtype=torch.float32
    )
    out_col = torch.empty((D,), device=arg1_1.device, dtype=torch.float32)
    out_f32 = torch.empty((32, 128, D), device=arg1_1.device, dtype=torch.float32)
    out_bf16 = torch.empty((M, D), device=arg1_1.device, dtype=torch.bfloat16)

    _mt5_residual_row_tile_kernel[(num_row_tiles,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        partial_col,
        out_f32,
        out_bf16,
        M_=M,
        D_=D,
        ROW_TILE=ROW_TILE,
        BLOCK_D=BLOCK_D,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        INV_D_=INV_D,
        num_warps=4,
        num_stages=3,
    )
    _finalize_column_sum_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_col,
        out_col,
        NUM_ROW_TILES=num_row_tiles,
        D_=D,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
        num_stages=3,
    )
    return out_col, out_f32, out_bf16, out_bf16.permute(1, 0)
