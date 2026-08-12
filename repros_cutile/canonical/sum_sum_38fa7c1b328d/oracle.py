"""cuTile port of sum_sum_38fa7c1b328d (COOPERATIVE_SPLIT_K): MT5 LN-backward
17-input residual chain, row reduction, column reduction.

Per row: sum 17 residual (bf16) inputs in fp32, mask/dropout, per-row LN-back
recomputation, column partial sum. Split into a producer (per-row-block) and
a finalizer that reduces column partials.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 4096
D = 512
BLOCK_D = 512
DROPOUT_SCALE = 1.1111111111111112
INV_D = 0.001953125


@ct.kernel
def _mt5_residual_row_tile_kernel(
    arg0_ptr, arg1_ptr, arg2_ptr, arg3_ptr, arg4_ptr,
    arg5_ptr, arg6_ptr, arg7_ptr, arg8_ptr, arg9_ptr,
    arg10_ptr, arg11_ptr, arg12_ptr, arg13_ptr, arg14_ptr,
    arg15_ptr, arg16_ptr,
    mask1_ptr,     # bool [M, D]
    weight_ptr,    # f32 [D]
    norm_arg_ptr,  # f32 [M, D]
    row_scale_ptr, # f32 [M]
    mask2_ptr,     # bool [M, D]
    partial_col_ptr, # f32 [num_row_tiles, D]  per-tile column partials
    out_f32_ptr,   # f32 [M, D]
    out_bf16_ptr,  # bf16 [M, D]
    ROW_TILE_C: ct.Constant[int],
    BLOCK_D_C: ct.Constant[int],
):
    tile_m = ct.bid(0)

    r = ct.load(arg1_ptr, index=(tile_m, 0), shape=(ROW_TILE_C, BLOCK_D_C))
    r_f = ct.astype(r, ct.float32)

    def add_arg(residual, ptr):
        v = ct.astype(ct.load(ptr, index=(tile_m, 0), shape=(ROW_TILE_C, BLOCK_D_C)), ct.float32)
        return residual + v

    r_f = add_arg(r_f, arg0_ptr)
    r_f = add_arg(r_f, arg2_ptr)
    r_f = add_arg(r_f, arg3_ptr)
    r_f = add_arg(r_f, arg4_ptr)
    r_f = add_arg(r_f, arg5_ptr)
    r_f = add_arg(r_f, arg6_ptr)
    r_f = add_arg(r_f, arg7_ptr)
    r_f = add_arg(r_f, arg8_ptr)
    r_f = add_arg(r_f, arg9_ptr)
    r_f = add_arg(r_f, arg10_ptr)
    r_f = add_arg(r_f, arg11_ptr)
    r_f = add_arg(r_f, arg12_ptr)
    r_f = add_arg(r_f, arg13_ptr)
    r_f = add_arg(r_f, arg14_ptr)
    r_f = add_arg(r_f, arg15_ptr)
    r_f = add_arg(r_f, arg16_ptr)

    mask1 = ct.astype(ct.load(mask1_ptr, index=(tile_m, 0), shape=(ROW_TILE_C, BLOCK_D_C)), ct.float32)
    norm_arg = ct.astype(ct.load(norm_arg_ptr, index=(tile_m, 0), shape=(ROW_TILE_C, BLOCK_D_C)), ct.float32)
    row_scale_tile = ct.astype(ct.load(row_scale_ptr, index=(tile_m,), shape=(ROW_TILE_C,)), ct.float32)
    row_scale = ct.reshape(row_scale_tile, (ROW_TILE_C, 1))

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_D_C,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_D_C))

    dropout_scale = mask1 * DROPOUT_SCALE
    masked_residual = r_f * dropout_scale
    weighted = masked_residual * weight_2d
    norm_scaled = norm_arg * row_scale

    partial_col = masked_residual * norm_scaled  # (ROW_TILE, BLOCK_D)

    row_terms = weighted * norm_arg
    row_sum = ct.sum(row_terms, axis=1)  # (ROW_TILE,)
    row_sum_2d = ct.reshape(row_sum, (ROW_TILE_C, 1))

    direct = weighted * row_scale
    row_scale_sq = row_scale * row_scale
    row_scale_cu = row_scale_sq * row_scale
    correction_scale = row_sum_2d * -0.5 * row_scale_cu * INV_D
    correction = correction_scale * (norm_arg * 2.0)
    add_value = direct + correction

    ct.store(out_f32_ptr, index=(tile_m, 0), tile=add_value)
    ct.store(partial_col_ptr, index=(tile_m, 0), tile=partial_col)

    add_bf16_f = ct.astype(ct.astype(add_value, ct.bfloat16), ct.float32)
    mask2 = ct.astype(ct.load(mask2_ptr, index=(tile_m, 0), shape=(ROW_TILE_C, BLOCK_D_C)), ct.float32)
    scaled_mask_bf16 = ct.astype(
        mask2 * DROPOUT_SCALE, ct.bfloat16,
    )
    out_bf16 = ct.astype(
        add_bf16_f * ct.astype(scaled_mask_bf16, ct.float32),
        ct.bfloat16,
    )
    ct.store(out_bf16_ptr, index=(tile_m, 0), tile=out_bf16)


@ct.kernel
def _finalize_column_sum_kernel(
    partial_col_ptr, # f32 [M, D]
    out_col_ptr,     # f32 [D]
    M_C: ct.Constant[int],
    FINAL_BLOCK_M_C: ct.Constant[int],
    FINAL_BLOCK_D_C: ct.Constant[int],
):
    col_tile = ct.bid(0)
    acc = ct.zeros((FINAL_BLOCK_D_C,), dtype=ct.float32)
    n_row_tiles = M_C // FINAL_BLOCK_M_C
    for r_block in range(n_row_tiles):
        p = ct.load(partial_col_ptr, index=(r_block, col_tile), shape=(FINAL_BLOCK_M_C, FINAL_BLOCK_D_C))
        acc = acc + ct.sum(p, axis=0)
    ct.store(out_col_ptr, index=(col_tile,), tile=acc)


@oracle_impl(
    hardware="B200",
    point="c4d61901",
    ROW_TILE=4,
    BLOCK_D=512,
    FINAL_BLOCK_D=8,
    FINAL_BLOCK_TILES=1024,
)
def oracle_forward(
    inputs, *, ROW_TILE, BLOCK_D, FINAL_BLOCK_D, FINAL_BLOCK_TILES,
):
    del FINAL_BLOCK_TILES
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
        arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16,
        arg17, arg18, arg19, arg20, arg21,
        *_shapes,
    ) = inputs

    device = arg1.device
    arg1_flat = arg1.view(M, D)
    arg17_flat = arg17.view(M, D)
    arg19_flat = arg19.view(M, D)
    arg20_flat = arg20.view(M)
    arg21_flat = arg21.view(M, D)

    partial_col = torch.empty((M, D), device=device, dtype=torch.float32)
    out_col = torch.empty((D,), device=device, dtype=torch.float32)
    out_f32 = torch.empty((32, 128, D), device=device, dtype=torch.float32)
    out_f32_2d = out_f32.view(M, D)
    out_bf16 = torch.empty((M, D), device=device, dtype=torch.bfloat16)

    num_row_tiles = M // ROW_TILE

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_tiles, 1, 1),
        _mt5_residual_row_tile_kernel,
        (
            arg0, arg1_flat, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
            arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16,
            arg17_flat, arg18, arg19_flat, arg20_flat, arg21_flat,
            partial_col, out_f32_2d, out_bf16, ROW_TILE, BLOCK_D,
        ),
    )
    ct.launch(
        stream,
        (D // FINAL_BLOCK_D, 1, 1),
        _finalize_column_sum_kernel,
        (partial_col, out_col, M, ROW_TILE, FINAL_BLOCK_D),
    )
    return out_col, out_f32, out_bf16, out_bf16.permute(1, 0)
