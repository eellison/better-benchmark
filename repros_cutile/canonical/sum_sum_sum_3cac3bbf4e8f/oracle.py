"""cuTile port of sum_sum_sum_3cac3bbf4e8f: LayerNorm-backward multi-stage.

Row-tiled [1152000, 512]. Per row:
  dy_gamma[c] = dy_scalar_f32 * gamma[c]
  normed[c] = (x_f[c] - mean_row) * invstd_row
  row_dot = sum_c(dy_gamma * normed)
  grad[c] = invstd_row/512 * (dy_gamma * 512 - sum_c(dy_gamma) - normed * row_dot)
  col_partial[c] += dy_scalar_f32 * normed[c]

Two-stage reduction on col_partial: [num_row_tiles, 512] -> [512].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 1_152_000
CHANNELS = 512
INV_C = 1.0 / CHANNELS
ROW_TILE = 64
NUM_ROW_TILES = ROWS // ROW_TILE  # 18000
PARTIAL_REDUCE_ROWS = 256  # rows-per-partial block for stage 1
FINAL_BLOCK_COLS = 16


@ct.kernel
def _ln_backward_row_kernel(
    invstd_ptr,      # f32 [ROWS, 1]
    dy_scalar_ptr,   # bf16 []  (loaded elsewhere; we load [1] slice)
    gamma_ptr,       # f32 [CHANNELS]
    x_ptr,           # bf16 [ROWS, CHANNELS]
    mean_ptr,        # f32 [ROWS, 1]
    out_grad_ptr,    # bf16 [ROWS, CHANNELS]
    partial_ptr,     # f32 [NUM_ROW_TILES, CHANNELS]
    ROW_TILE_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    INV_C_: ct.Constant[float],
):
    tile = ct.bid(0)

    # Load dy_scalar as [1] bf16 (reshape done in oracle_forward).
    dy_scalar_bf = ct.load(dy_scalar_ptr, index=(0,), shape=(1,))
    dy_scalar_f = ct.astype(dy_scalar_bf, ct.float32)
    dy_scalar_val = dy_scalar_f  # (1,)
    # gamma [CHANNELS] fp32
    gamma = ct.load(gamma_ptr, index=(0,), shape=(CHANNELS_,))
    dy_gamma = dy_scalar_val * gamma  # (CHANNELS,) via broadcast
    dy_gamma_sum = ct.sum(dy_gamma)  # scalar

    # Load ROW_TILE rows.
    # x is [ROWS, CHANNELS] bf16
    x_bf = ct.load(x_ptr, index=(tile, 0), shape=(ROW_TILE_, CHANNELS_))
    x_f = ct.astype(x_bf, ct.float32)
    # mean, invstd are [ROWS, 1] fp32 → load (ROW_TILE, 1)
    mean_tile = ct.load(mean_ptr, index=(tile, 0), shape=(ROW_TILE_, 1))
    invstd_tile = ct.load(invstd_ptr, index=(tile, 0), shape=(ROW_TILE_, 1))

    centered = x_f - mean_tile  # (ROW_TILE, CHANNELS) via broadcast on last dim
    normed = centered * invstd_tile  # (ROW_TILE, CHANNELS)

    # Per row: row_dot = sum_c(dy_gamma * normed)
    dy_gamma_2d = ct.reshape(dy_gamma, (1, CHANNELS_))
    dy_gn = dy_gamma_2d * normed  # broadcast (ROW_TILE, CHANNELS)
    row_dot = ct.sum(dy_gn, axis=1, keepdims=True)  # (ROW_TILE, 1)

    # grad = invstd/C * (dy_gamma * C - dy_gamma_sum - normed * row_dot)
    dy_gamma_scaled_2d = dy_gamma_2d * float(CHANNELS_)  # (1, CHANNELS)
    scale = invstd_tile * INV_C_  # (ROW_TILE, 1)
    grad = scale * (dy_gamma_scaled_2d - dy_gamma_sum - normed * row_dot)
    grad_bf = ct.astype(grad, ct.bfloat16)
    ct.store(out_grad_ptr, index=(tile, 0), tile=grad_bf)

    # col_partial for this tile: sum_over_rows(dy_scalar_f * normed[r, c])
    col_partial_tile = ct.sum(dy_scalar_val * normed, axis=0)  # (CHANNELS,)
    ct.store(partial_ptr, index=(tile, 0), tile=ct.reshape(col_partial_tile, (1, CHANNELS_)))


@ct.kernel
def _reduce_partial_kernel(
    partial_ptr,   # f32 [NUM_ROW_TILES, CHANNELS]
    stage_ptr,     # f32 [NUM_STAGE_TILES, CHANNELS]
    NUM_ROW_TILES_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    tile_block = ct.bid(0)
    col_block = ct.bid(1)
    vals = ct.load(
        partial_ptr,
        index=(tile_block, col_block),
        shape=(BLOCK_TILES, BLOCK_COLS),
        padding_mode=ct.PaddingMode.ZERO,
    )
    reduced = ct.sum(vals, axis=0)  # (BLOCK_COLS,)
    ct.store(stage_ptr, index=(tile_block, col_block),
             tile=ct.reshape(reduced, (1, BLOCK_COLS)))


@ct.kernel
def _final_reduce_kernel(
    stage_ptr,     # f32 [NUM_STAGE_TILES, CHANNELS]
    out_sum_ptr,   # f32 [CHANNELS]
    NUM_STAGE_TILES_PAD: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    vals = ct.load(
        stage_ptr,
        index=(0, col_block),
        shape=(NUM_STAGE_TILES_PAD, BLOCK_COLS),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(vals, axis=0)  # (BLOCK_COLS,)
    ct.store(out_sum_ptr, index=(col_block,), tile=total)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="4e21884e")
def oracle_forward(inputs):
    invstd, dy_scalar, gamma, x, mean, shape_param = inputs
    device = x.device

    out_grad = torch.empty_strided(
        tuple(int(d) for d in shape_param),
        (CHANNELS, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # Views
    invstd_2d = invstd.view(ROWS, 1)
    mean_2d = mean.view(ROWS, 1)
    x_2d = x.view(ROWS, CHANNELS)
    out_grad_2d = out_grad.view(ROWS, CHANNELS)

    partial = torch.empty((NUM_ROW_TILES, CHANNELS), device=device, dtype=torch.float32)
    num_stage_tiles = (NUM_ROW_TILES + PARTIAL_REDUCE_ROWS - 1) // PARTIAL_REDUCE_ROWS
    # Round up to power of 2 for the second-stage tile shape.
    num_stage_tiles_pad = _next_pow2(num_stage_tiles)
    stage = torch.zeros((num_stage_tiles_pad, CHANNELS), device=device, dtype=torch.float32)
    out_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    # Reshape dy_scalar from [] to [1] for cuTile.
    dy_scalar_1d = dy_scalar.view(1)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (NUM_ROW_TILES, 1, 1),
        _ln_backward_row_kernel,
        (invstd_2d, dy_scalar_1d, gamma, x_2d, mean_2d,
         out_grad_2d, partial,
         ROW_TILE, CHANNELS, INV_C),
    )

    # Stage 1: reduce partial [NUM_ROW_TILES, CHANNELS] with block (PARTIAL_REDUCE_ROWS, FINAL_BLOCK_COLS)
    ct.launch(
        stream,
        (num_stage_tiles, CHANNELS // FINAL_BLOCK_COLS, 1),
        _reduce_partial_kernel,
        (partial, stage, NUM_ROW_TILES, CHANNELS,
         PARTIAL_REDUCE_ROWS, FINAL_BLOCK_COLS),
    )

    # Stage 2: reduce stage [num_stage_tiles_pad, CHANNELS] -> out_sum [CHANNELS]
    ct.launch(
        stream,
        (CHANNELS // FINAL_BLOCK_COLS, 1, 1),
        _final_reduce_kernel,
        (stage, out_sum, num_stage_tiles_pad, FINAL_BLOCK_COLS),
    )
    return out_grad, out_sum
