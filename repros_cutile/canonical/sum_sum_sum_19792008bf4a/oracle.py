"""cuTile port of sum_sum_sum_19792008bf4a: BERT layer-norm backward with
cooperative multi-output split-K reduction.

Three sequential kernels:
1. `_row_stats_kernel` — per-row scan over hidden, produces per-row
   `row_neg_x_gamma_sum` and `row_coef` vectors.
2. `_materialize_and_partial_sum_kernel` — per (row_block, col_block) tile:
   compute f32 `add_3` output, bf16 dropout-masked output, and produce
   per-tile column partials for three column reductions.
3. `_finalize_column_sums_kernel` — sum column partials over row-tiles and
   emit the three (hidden,) column outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
EPS = 1.0e-6
INV_HIDDEN = 1.0 / HIDDEN
ROW_BACKWARD_SCALE = 0.002607561929595828
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _row_stats_kernel(
    x_ptr,                   # bf16 [ROWS, HIDDEN]
    gamma_ptr,               # f32  [HIDDEN]
    dy_ptr,                  # f32  [ROWS, HIDDEN]
    denom_base_ptr,          # f32  [ROWS]
    full_ptr,                # f32  scalar
    row_neg_x_gamma_sum_ptr, # f32  [ROWS] out
    row_coef_ptr,            # f32  [ROWS] out
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row_block = ct.bid(0)
    row_off = ct.arange(ROW_BLOCK, dtype=ct.int32) + row_block * ROW_BLOCK
    col_off = ct.arange(BLOCK_H, dtype=ct.int32)
    row_valid = row_off < ROWS_
    col_valid = col_off < HIDDEN_
    mask2 = ct.reshape(row_valid, (ROW_BLOCK, 1)) & ct.reshape(col_valid, (1, BLOCK_H))
    row_tile = ct.reshape(row_off, (ROW_BLOCK, 1)) + ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.int32)
    col_tile = ct.reshape(col_off, (1, BLOCK_H)) + ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.int32)

    x_bf = ct.gather(x_ptr, (row_tile, col_tile), mask=mask2, padding_value=0)
    x = ct.astype(x_bf, ct.float32)
    dy_f = ct.astype(
        ct.gather(dy_ptr, (row_tile, col_tile), mask=mask2, padding_value=0),
        ct.float32,
    )
    gamma = ct.astype(
        ct.gather(gamma_ptr, col_off, mask=col_valid, padding_value=0),
        ct.float32,
    )
    denom_base_row = ct.astype(
        ct.gather(denom_base_ptr, row_off, mask=row_valid, padding_value=0),
        ct.float32,
    )
    inv_denom = 1.0 / (denom_base_row + EPS)
    inv_denom_2d = ct.reshape(inv_denom, (ROW_BLOCK, 1)) + ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)
    gamma_2d = ct.reshape(gamma, (1, BLOCK_H)) + ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)

    x_gamma_over_denom = (x * inv_denom_2d) * gamma_2d
    zeros2 = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)
    curv_val = ct.where(mask2, x_gamma_over_denom * dy_f * inv_denom_2d, zeros2)
    row_curvature_sum = -ct.sum(curv_val, axis=1)
    row_neg = ct.sum(ct.where(mask2, -x_gamma_over_denom, zeros2), axis=1)

    full_scalar = ct.load(full_ptr, index=(0,), shape=(1,))
    full_val = ct.astype(full_scalar, ct.float32)
    full_bcast = ct.reshape(full_val, (1,)) + ct.zeros((ROW_BLOCK,), dtype=ct.float32)

    zeros_r = ct.zeros((ROW_BLOCK,), dtype=ct.float32)
    row_coef_val = ct.where(
        denom_base_row == zeros_r,
        full_bcast,
        row_curvature_sum / (denom_base_row * 2.0),
    )
    row_coef_val = row_coef_val * ROW_BACKWARD_SCALE

    ct.scatter(row_neg_x_gamma_sum_ptr, row_off, row_neg, mask=row_valid)
    ct.scatter(row_coef_ptr, row_off, row_coef_val, mask=row_valid)


@ct.kernel
def _materialize_and_partial_sum_kernel(
    x_ptr,                # bf16 [ROWS, HIDDEN]
    gamma_ptr,            # f32  [HIDDEN]
    dy_ptr,               # f32  [ROWS, HIDDEN]
    denom_base_ptr,       # f32  [ROWS]
    residual_ptr,         # f32  [ROWS, HIDDEN]
    keep_ptr,             # bool [ROWS, HIDDEN]
    row_neg_ptr,          # f32  [ROWS]
    row_coef_ptr,         # f32  [ROWS]
    partial_x_ptr,        # f32  [NUM_TILES, HIDDEN]
    partial_x_dy_ptr,     # f32  [NUM_TILES, HIDDEN]
    partial_out_ptr,      # f32  [NUM_TILES, HIDDEN]
    add_out_ptr,          # f32  [ROWS, HIDDEN]
    bf16_out_ptr,         # bf16 [ROWS, HIDDEN]
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)
    row_off = ct.arange(ROW_BLOCK, dtype=ct.int32) + row_block * ROW_BLOCK
    col_off = ct.arange(BLOCK_C, dtype=ct.int32) + col_block * BLOCK_C
    row_valid = row_off < ROWS_
    col_valid = col_off < HIDDEN_
    mask2 = ct.reshape(row_valid, (ROW_BLOCK, 1)) & ct.reshape(col_valid, (1, BLOCK_C))
    row_tile = ct.reshape(row_off, (ROW_BLOCK, 1)) + ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.int32)
    col_tile = ct.reshape(col_off, (1, BLOCK_C)) + ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.int32)

    x_bf = ct.gather(x_ptr, (row_tile, col_tile), mask=mask2, padding_value=0)
    x = ct.astype(x_bf, ct.float32)
    dy_f = ct.astype(
        ct.gather(dy_ptr, (row_tile, col_tile), mask=mask2, padding_value=0),
        ct.float32,
    )
    gamma = ct.astype(
        ct.gather(gamma_ptr, col_off, mask=col_valid, padding_value=0),
        ct.float32,
    )
    residual = ct.astype(
        ct.gather(residual_ptr, (row_tile, col_tile), mask=mask2, padding_value=0),
        ct.float32,
    )
    denom_base_row = ct.astype(
        ct.gather(denom_base_ptr, row_off, mask=row_valid, padding_value=0),
        ct.float32,
    )
    row_coef = ct.astype(
        ct.gather(row_coef_ptr, row_off, mask=row_valid, padding_value=0),
        ct.float32,
    )
    row_neg = ct.astype(
        ct.gather(row_neg_ptr, row_off, mask=row_valid, padding_value=0),
        ct.float32,
    )
    keep = ct.astype(
        ct.gather(keep_ptr, (row_tile, col_tile), mask=mask2, padding_value=0),
        ct.float32,
    )

    inv_denom = 1.0 / (denom_base_row + EPS)
    inv_denom_2d = ct.reshape(inv_denom, (ROW_BLOCK, 1)) + ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)
    gamma_2d = ct.reshape(gamma, (1, BLOCK_C)) + ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)
    row_coef_2d = ct.reshape(row_coef, (ROW_BLOCK, 1)) + ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)
    row_neg_2d = ct.reshape(row_neg, (ROW_BLOCK, 1)) + ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)

    x_over_denom = x * inv_denom_2d
    x_gamma_over_denom = x_over_denom * gamma_2d
    add_1 = residual + x_gamma_over_denom
    add_2 = add_1 + row_coef_2d * dy_f
    add_3 = add_2 + row_neg_2d * INV_HIDDEN

    ct.scatter(add_out_ptr, (row_tile, col_tile), add_3, mask=mask2)

    keep_bf16 = ct.astype(keep, ct.bfloat16)
    keep_scaled = ct.astype(ct.astype(keep_bf16, ct.float32) * DROPOUT_SCALE,
                            ct.bfloat16)
    keep_scaled_f = ct.astype(keep_scaled, ct.float32)
    add_3_bf16 = ct.astype(add_3, ct.bfloat16)
    add_3_bf16_f = ct.astype(add_3_bf16, ct.float32)
    bf16_out = ct.astype(add_3_bf16_f * keep_scaled_f, ct.bfloat16)
    ct.scatter(bf16_out_ptr, (row_tile, col_tile), bf16_out, mask=mask2)

    zeros2 = ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)
    sum_x = ct.sum(ct.where(mask2, x, zeros2), axis=0)
    sum_x_dy = ct.sum(ct.where(mask2, x_over_denom * dy_f, zeros2), axis=0)
    sum_out = ct.sum(ct.where(mask2, ct.astype(bf16_out, ct.float32), zeros2), axis=0)

    tile_row = ct.full((BLOCK_C,), row_block, dtype=ct.int32)
    ct.scatter(partial_x_ptr, (tile_row, col_off), sum_x, mask=col_valid)
    ct.scatter(partial_x_dy_ptr, (tile_row, col_off), sum_x_dy, mask=col_valid)
    ct.scatter(partial_out_ptr, (tile_row, col_off), sum_out, mask=col_valid)


@ct.kernel
def _finalize_column_sums_kernel(
    partial_x_ptr,
    partial_x_dy_ptr,
    partial_out_ptr,
    out_x_ptr,
    out_x_dy_ptr,
    out_out_ptr,
    NUM_TILES: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    col_off = ct.arange(BLOCK_C, dtype=ct.int32) + col_block * BLOCK_C
    col_valid = col_off < HIDDEN_

    acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_x_dy = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_out = ct.zeros((BLOCK_C,), dtype=ct.float32)

    num_iter = (NUM_TILES + BLOCK_TILES - 1) // BLOCK_TILES
    for i in range(num_iter):
        tile_off = ct.arange(BLOCK_TILES, dtype=ct.int32) + i * BLOCK_TILES
        tile_valid = tile_off < NUM_TILES
        mask2 = ct.reshape(tile_valid, (BLOCK_TILES, 1)) & ct.reshape(col_valid, (1, BLOCK_C))
        tile_tile = ct.reshape(tile_off, (BLOCK_TILES, 1)) + ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.int32)
        col_tile = ct.reshape(col_off, (1, BLOCK_C)) + ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.int32)
        px = ct.astype(ct.gather(partial_x_ptr, (tile_tile, col_tile),
                                  mask=mask2, padding_value=0), ct.float32)
        pxdy = ct.astype(ct.gather(partial_x_dy_ptr, (tile_tile, col_tile),
                                    mask=mask2, padding_value=0), ct.float32)
        pout = ct.astype(ct.gather(partial_out_ptr, (tile_tile, col_tile),
                                    mask=mask2, padding_value=0), ct.float32)
        zeros2 = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)
        acc_x = acc_x + ct.sum(ct.where(mask2, px, zeros2), axis=0)
        acc_x_dy = acc_x_dy + ct.sum(ct.where(mask2, pxdy, zeros2), axis=0)
        acc_out = acc_out + ct.sum(ct.where(mask2, pout, zeros2), axis=0)

    ct.scatter(out_x_ptr, col_off, acc_x, mask=col_valid)
    ct.scatter(out_x_dy_ptr, col_off, acc_x_dy, mask=col_valid)
    # Round to bf16, then back to f32 for the bf16-sum output
    acc_out_rt = ct.astype(ct.astype(acc_out, ct.bfloat16), ct.float32)
    ct.scatter(out_out_ptr, col_off, acc_out_rt, mask=col_valid)


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
)
def oracle_forward(
    inputs,
    *,
    STATS_ROW_BLOCK,
    STATS_BLOCK_H,
    ELEM_ROW_BLOCK,
    ELEM_BLOCK_C,
    FINAL_BLOCK_C,
    FINAL_BLOCK_TILES,
):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     _shape_param_0, _shape_param_1, _shape_param_2,
     _shape_param_3, _shape_param_4, _shape_param_5) = inputs

    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    device = arg0_1.device

    # dy and residual come as 3D contiguous [16, 128, 768]; flatten to [rows, hidden]
    dy_2d = arg2_1.reshape(rows, hidden)
    denom_1d = arg3_1.reshape(rows)
    residual_2d = arg4_1.reshape(rows, hidden)
    keep_2d = arg6_1.reshape(rows, hidden)

    row_neg = torch.empty((rows,), device=device, dtype=torch.float32)
    row_coef = torch.empty((rows,), device=device, dtype=torch.float32)

    num_row_tiles = (rows + ELEM_ROW_BLOCK - 1) // ELEM_ROW_BLOCK
    partial_x = torch.empty((num_row_tiles, hidden), device=device, dtype=torch.float32)
    partial_x_dy = torch.empty((num_row_tiles, hidden), device=device, dtype=torch.float32)
    partial_out = torch.empty((num_row_tiles, hidden), device=device, dtype=torch.float32)

    add_shape = tuple(int(d) for d in _shape_param_3)
    bf16_shape = tuple(int(d) for d in _shape_param_4)
    add_out_2d = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    bf16_out_2d = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    num_stats_tiles = (rows + STATS_ROW_BLOCK - 1) // STATS_ROW_BLOCK
    ct.launch(
        stream,
        (num_stats_tiles, 1, 1),
        _row_stats_kernel,
        (arg0_1, arg1_1, dy_2d, denom_1d, arg5_1.reshape(1),
         row_neg, row_coef, rows, hidden, STATS_ROW_BLOCK, STATS_BLOCK_H),
    )
    ct.launch(
        stream,
        (num_row_tiles, (hidden + ELEM_BLOCK_C - 1) // ELEM_BLOCK_C, 1),
        _materialize_and_partial_sum_kernel,
        (arg0_1, arg1_1, dy_2d, denom_1d, residual_2d, keep_2d,
         row_neg, row_coef, partial_x, partial_x_dy, partial_out,
         add_out_2d, bf16_out_2d,
         rows, hidden, ELEM_ROW_BLOCK, ELEM_BLOCK_C),
    )

    sum_x = torch.empty((hidden,), device=device, dtype=torch.float32)
    sum_x_dy = torch.empty((hidden,), device=device, dtype=torch.float32)
    sum_out = torch.empty((hidden,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        ((hidden + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_column_sums_kernel,
        (partial_x, partial_x_dy, partial_out,
         sum_x, sum_x_dy, sum_out,
         num_row_tiles, hidden, FINAL_BLOCK_TILES, FINAL_BLOCK_C),
    )

    add_out = add_out_2d.view(add_shape)
    bf16_out = bf16_out_2d.view(bf16_shape)
    view_3 = bf16_out.view(bf16_shape)
    return (
        sum_x.view(tuple(int(d) for d in _shape_param_1)),
        sum_x_dy.view(tuple(int(d) for d in _shape_param_2)),
        add_out,
        view_3,
        view_3.permute(1, 0),
        sum_out.view(tuple(int(d) for d in _shape_param_5)),
    )
