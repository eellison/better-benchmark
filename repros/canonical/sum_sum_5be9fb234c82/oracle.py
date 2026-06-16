"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MT5 bf16 dropout/RMSNorm-backward scope by row-tiling the `[4096, 512]` producer, sharing the row-local reduction for the f32 backward tensor and bf16 masked output, emitting partial column sums for the returned `[512]` vector, and returning the required flat output plus transpose alias from one backing allocation, whereas Inductor schedules the row reduction, column reduction, bf16 epilogue, and layout-only alias as separate generic regions over materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local reductions, compatible column partials, dtype-cast epilogues, and alias-preserving layout stores; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer that fuses the RMSNorm-backward epilogue, writes the visible side outputs, emits column partials, and finalizes the side vector."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
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
def _row_tile_kernel(
    x_ptr,
    mask1_ptr,
    weight_ptr,
    arg_ptr,
    row_scale_ptr,
    mask2_ptr,
    partial_col_ptr,
    add_out_ptr,
    bf16_out_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    INV_D: tl.constexpr,
    ROW_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_m = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    partial_col = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mask_scale_bf16 = tl.full((BLOCK_D,), SCALE, tl.float32).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    for row_i in tl.range(0, ROW_TILE):
        row = tile_m * ROW_TILE + row_i
        mask = (row < M) & col_mask
        offsets = row * D + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mask1 = tl.load(mask1_ptr + offsets, mask=mask, other=0).to(tl.float32)
        arg = tl.load(arg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row < M, other=0.0).to(tl.float32)

        dropout_scale = _mul_rn_f32(mask1, SCALE)
        masked_x = _mul_rn_f32(x, dropout_scale)
        weighted = _mul_rn_f32(masked_x, weight)
        arg_scaled = _mul_rn_f32(arg, row_scale)
        partial_col = _add_rn_f32(
            partial_col,
            tl.where(mask, _mul_rn_f32(masked_x, arg_scaled), 0.0),
        )

        row_sum_values = _mul_rn_f32(weighted, arg)
        row_sum = tl.sum(tl.where(mask, row_sum_values, 0.0), axis=0)

        row_scale_sq = _mul_rn_f32(row_scale, row_scale)
        row_scale_cu = _mul_rn_f32(row_scale_sq, row_scale)
        correction_scale = _mul_rn_f32(_mul_rn_f32(row_sum, -0.5), row_scale_cu)
        correction_scale = _mul_rn_f32(correction_scale, INV_D)
        correction = _mul_rn_f32(correction_scale, _mul_rn_f32(arg, 2.0))
        direct = _mul_rn_f32(weighted, row_scale)
        add_value = _add_rn_f32(direct, correction)
        tl.store(add_out_ptr + offsets, add_value, mask=mask)

        add_bf16 = add_value.to(tl.bfloat16, fp_downcast_rounding="rtne")
        mask2 = tl.load(mask2_ptr + offsets, mask=mask, other=0)
        out_bf16 = _mul_rn_f32(
            add_bf16.to(tl.float32),
            tl.where(mask2, mask_scale_bf16, tl.full((BLOCK_D,), 0.0, tl.bfloat16)).to(tl.float32),
        ).to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(bf16_out_ptr + offsets, out_bf16, mask=mask)

    tl.store(partial_col_ptr + tile_m * D + cols, partial_col, mask=col_mask)


@triton.jit
def _finalize_column_sum_kernel(
    partial_col_ptr,
    out_col_ptr,
    NUM_ROW_TILES: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < D
    tiles = tl.arange(0, BLOCK_TILES)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for tile_start in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tile_ids = tile_start + tiles
        mask = (tile_ids[:, None] < NUM_ROW_TILES) & col_mask[None, :]
        offsets = tile_ids[:, None] * D + cols[None, :]
        values = tl.load(partial_col_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(values, axis=0)

    tl.store(out_col_ptr + cols, acc, mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="8b6fe472",
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
        x,
        mask1,
        weight,
        arg,
        row_scale,
        mask2,
        _shape0,
        _shape1,
        _shape2,
        _shape3,
    ) = inputs

    m, d = (int(dim) for dim in x.shape)
    num_row_tiles = triton.cdiv(m, ROW_TILE)
    partial_col = torch.empty((num_row_tiles, d), device=x.device, dtype=torch.float32)
    col_out = torch.empty((d,), device=x.device, dtype=torch.float32)
    add_out = torch.empty((32, 128, d), device=x.device, dtype=torch.float32)
    bf16_out = torch.empty_strided(tuple(x.shape), tuple(int(s) for s in x.stride()), device=x.device, dtype=torch.bfloat16)

    _row_tile_kernel[(num_row_tiles,)](
        x,
        mask1,
        weight,
        arg,
        row_scale,
        mask2,
        partial_col,
        add_out,
        bf16_out,
        M=m,
        D=d,
        SCALE=1.1111111111111112,
        INV_D=1.0 / 512.0,
        ROW_TILE=ROW_TILE,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )
    _finalize_column_sum_kernel[(triton.cdiv(d, FINAL_BLOCK_D),)](
        partial_col,
        col_out,
        NUM_ROW_TILES=num_row_tiles,
        D=d,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
        num_stages=3,
    )
    return col_out, add_out, bf16_out, bf16_out.permute(1, 0)
