"""cuTile port of sum_sum_4c6ae5dbcf21 (COOPERATIVE_SPLIT_K): MT5/T5
LayerNorm-backward via a row-tile producer + column-sum finalize pair,
mirroring the Triton _row_tile_kernel / _finalize_column_sum_kernel.

Each row-tile program loads a row-shared weight tile, then loops ROW_TILE
times to load per-row x_bf16 / rhs / row_scale / residual / keep, does the
row-local sum via ct.sum(axis=0) (scalar), computes the LN-backward
epilogue in-kernel, stores the f32 add_out and the bf16 dropout-scaled
output, and accumulates a per-tile column partial. A second kernel reduces
the partials over tiles (with padding_mode=ZERO for the power-of-two
tile-height) to the final f32 [D] column sum. cuTile's default RTNE
downcasts and round-to-nearest fp32 arithmetic match Triton's inline PTX
add.rn/mul.rn.
"""
from __future__ import annotations

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


D = 512
DROP_SCALE = 1.1111111111111112
INV_D = 1.0 / D
BLOCK_D = 512


@ct.kernel
def _row_tile_kernel(
    x_bf16_ptr,       # bf16 [M, D]
    weight_ptr,       # f32 [D]
    rhs_ptr,          # f32 [M, D]
    row_scale_ptr,    # f32 [M]
    residual_ptr,     # f32 [M, D]
    keep_ptr,         # bool [M, D]
    partial_col_ptr,  # f32 [num_row_tiles, D]
    add_out_ptr,      # f32 [M, D]
    bf16_out_ptr,     # bf16 [M, D]
    ROW_TILE: ct.Constant[int],
    BLOCK_D_: ct.Constant[int],
    DROP_SCALE_: ct.Constant[float],
    INV_D_: ct.Constant[float],
):
    tile_m = ct.bid(0)
    # Weight is shared across all ROW_TILE rows in this program.
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_D_,))  # (D,) f32
    partial_col = ct.zeros((BLOCK_D_,), dtype=ct.float32)

    for row_i in range(ROW_TILE):
        row = tile_m * ROW_TILE + row_i

        x_bf = ct.reshape(
            ct.load(x_bf16_ptr, index=(row, 0), shape=(1, BLOCK_D_)),
            (BLOCK_D_,),
        )
        x = ct.astype(x_bf, ct.float32)
        rhs = ct.reshape(
            ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_D_)),
            (BLOCK_D_,),
        )
        residual = ct.reshape(
            ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_D_)),
            (BLOCK_D_,),
        )
        keep_b = ct.reshape(
            ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_D_)),
            (BLOCK_D_,),
        )
        keep_f = ct.astype(keep_b, ct.float32)
        # row_scale kept as (1,) tile: broadcasts naturally with (BLOCK_D,).
        row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))

        weighted = x * weight                       # (BLOCK_D,)
        rhs_scaled = rhs * row_scale                # (BLOCK_D,) * (1,) -> (BLOCK_D,)
        partial_col = partial_col + x * rhs_scaled

        # Row-local reduction (in-kernel; fairness with Triton's tl.sum).
        row_terms = weighted * rhs
        row_sum_scalar = ct.sum(row_terms, axis=0)
        row_sum_1d = ct.reshape(row_sum_scalar, (1,))

        direct = weighted * row_scale               # (BLOCK_D,)
        add = residual + direct

        row_scale_sq = row_scale * row_scale
        row_scale_cu = row_scale_sq * row_scale
        scaled_sum = row_sum_1d * -0.5 * row_scale_cu   # (1,)
        div_v = scaled_sum * INV_D_                     # (1,)
        correction = div_v * (rhs * 2.0)                # (1,) * (BLOCK_D,) -> (BLOCK_D,)
        add_out = add + correction

        # bf16 round-trip boundary matches Triton's convert_element_type
        # / .to(tl.bfloat16) chain.
        add_bf16 = ct.astype(add_out, ct.bfloat16)
        keep_scale_bf = ct.astype(keep_f * DROP_SCALE_, ct.bfloat16)
        bf16_out = ct.astype(
            ct.astype(add_bf16, ct.float32)
            * ct.astype(keep_scale_bf, ct.float32),
            ct.bfloat16,
        )

        ct.store(add_out_ptr, index=(row, 0),
                 tile=ct.reshape(add_out, (1, BLOCK_D_)))
        ct.store(bf16_out_ptr, index=(row, 0),
                 tile=ct.reshape(bf16_out, (1, BLOCK_D_)))

    ct.store(partial_col_ptr, index=(tile_m, 0),
             tile=ct.reshape(partial_col, (1, BLOCK_D_)))


@ct.kernel
def _finalize_column_sum_kernel(
    partial_col_ptr,  # f32 [num_row_tiles, D]
    out_col_ptr,      # f32 [D]
    NUM_ROW_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(
        partial_col_ptr, index=(0, col_block),
        shape=(BLOCK_TILES, BLOCK_COLS),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < NUM_ROW_TILES, (BLOCK_TILES, 1))
    zero = ct.zeros((BLOCK_TILES, BLOCK_COLS), dtype=ct.float32)
    valid = ct.where(row_mask, values, zero)
    reduced = ct.sum(valid, axis=0)  # (BLOCK_COLS,)
    ct.store(out_col_ptr, index=(col_block,), tile=reduced)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_p2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


# 1c935b51: (T([4096,512], bf16), T([512], f32), T([32,128,512], f32), ...)
# a4c9a272: (T([8192,512], bf16), T([512], f32), T([8,1024,512], f32), ...)
@oracle_impl(hardware="B200", point="1c935b51", ROW_TILE=4, FINAL_BLOCK_COLS=8)
@oracle_impl(hardware="B200", point="a4c9a272", ROW_TILE=4, FINAL_BLOCK_COLS=8)
def oracle_forward(inputs, *, ROW_TILE: int, FINAL_BLOCK_COLS: int):
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
    num_row_tiles = m // ROW_TILE
    block_tiles = _next_p2(num_row_tiles)

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
    partial_col = torch.empty(
        (num_row_tiles, D), device=x_bf16.device, dtype=torch.float32,
    )
    out_col = torch.empty_strided(
        _shape_tuple(sum_shape_param),
        (1,),
        device=x_bf16.device,
        dtype=torch.float32,
    )

    # Flatten the 3D inputs into (M, D) / (M,) tile-space views for
    # tile-space indexing. All inputs are contiguous so .view/.reshape is
    # cost-free.
    rhs_2d = rhs.reshape(m, D)
    residual_2d = residual.reshape(m, D)
    keep_2d = keep.reshape(m, D)
    row_scale_1d = row_scale.reshape(m)
    add_out_2d = add_out.view(m, D)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_tiles, 1, 1), _row_tile_kernel,
        (x_bf16, weight, rhs_2d, row_scale_1d, residual_2d, keep_2d,
         partial_col, add_out_2d, bf16_out,
         ROW_TILE, BLOCK_D, DROP_SCALE, INV_D),
    )
    ct.launch(
        stream, ((D + FINAL_BLOCK_COLS - 1) // FINAL_BLOCK_COLS, 1, 1),
        _finalize_column_sum_kernel,
        (partial_col, out_col, num_row_tiles, block_tiles, FINAL_BLOCK_COLS),
    )
    return out_col, add_out, bf16_out, bf16_out.permute(1, 0)
