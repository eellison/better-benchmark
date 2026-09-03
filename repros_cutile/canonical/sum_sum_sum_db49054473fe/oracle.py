"""cuTile port of sum_sum_sum_db49054473fe: DINOv2 LN backward + residual + affine.

Structure follows the Triton reference's 3-kernel plan:
  Kernel 1 (_row_reductions_kernel): per (BLOCK_R, BLOCK_C) tile -> row_sum, row_dot
     (row-level reductions of gamma-weighted x and gamma-weighted x * norm).
  Kernel 2 (_epilogue_reduce_kernel): per (TILE_ROWS, TILE_COLS) tile:
     - compute grad, add (= residual + grad), side (bf16-rounded side output).
     - store add_out and side_out.
     - accumulate column partials for out0, out1, out3, out6 into
       partial[4, NUM_TILES, CHANNELS].
  Kernel 3 (_finalize_kernel): reduce partials along num_tiles to get out0, out1, out3, out6
     (with bf16 round for out6).

D_PAD = 1024 accommodates HIDDEN=768 in power-of-2 tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
CHANNELS = 768
ROWS = BATCH * TOKENS
ROW_BLOCK = 16
ROW_C_BLOCK = 1024   # pow2 >= CHANNELS
TILE_ROWS = 64
TILE_COLS = 64
FINAL_BLOCK_T = 4096
ADD_SHAPE = (BATCH, TOKENS, CHANNELS)
ADD_STRIDE = (TOKENS * CHANNELS, CHANNELS, 1)
SIDE_SHAPE = (ROWS, CHANNELS)
SIDE_STRIDE = (CHANNELS, 1)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


NUM_TILES = _cdiv(ROWS, TILE_ROWS)


@ct.kernel
def _row_reductions_kernel(
    x_ptr,           # bf16 [ROWS, CHANNELS_PAD]
    gamma_ptr,       # f32  [CHANNELS_PAD]
    norm_ptr,        # f32  [ROWS, CHANNELS_PAD]
    row_sum_ptr,     # f32  [ROWS]
    row_dot_ptr,     # f32  [ROWS]
    CHANNELS_ACTUAL: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile_r = ct.bid(0)

    # Column mask.
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask_1d = col_idx < CHANNELS_ACTUAL
    col_mask_2d = ct.reshape(col_mask_1d, (1, BLOCK_C))

    zero_1d = ct.zeros((BLOCK_C,), dtype=ct.float32)
    zero_2d = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)

    # Load gamma[BLOCK_C] and mask.
    gamma = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_C,))
    gamma = ct.where(col_mask_1d, gamma, zero_1d)
    gamma_2d = ct.reshape(gamma, (1, BLOCK_C))

    x = ct.astype(
        ct.load(x_ptr, index=(tile_r, 0), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    norm = ct.load(norm_ptr, index=(tile_r, 0), shape=(BLOCK_R, BLOCK_C))

    weighted = x * gamma_2d
    weighted_m = ct.where(col_mask_2d, weighted, zero_2d)
    row_sum = ct.sum(weighted_m, axis=1)  # (BLOCK_R,)
    weighted_dot_m = ct.where(col_mask_2d, weighted * norm, zero_2d)
    row_dot = ct.sum(weighted_dot_m, axis=1)  # (BLOCK_R,)

    ct.store(row_sum_ptr, index=(tile_r,), tile=row_sum)
    ct.store(row_dot_ptr, index=(tile_r,), tile=row_dot)


@ct.kernel
def _epilogue_reduce_kernel(
    x_ptr,           # bf16 [ROWS, CHANNELS_PAD]
    gamma_ptr,       # f32  [CHANNELS_PAD]
    norm_ptr,        # f32  [ROWS, CHANNELS_PAD]
    scale_ptr,       # f32  [BATCH * 1376]
    residual_ptr,    # f32  [ROWS, CHANNELS_PAD]
    side_scale_ptr,  # f32  [CHANNELS_PAD]
    side_in_ptr,     # bf16 [ROWS, CHANNELS_PAD]
    row_sum_ptr,     # f32  [ROWS]
    row_dot_ptr,     # f32  [ROWS]
    add_out_ptr,     # f32  [ROWS, CHANNELS_PAD]
    side_out_ptr,    # bf16 [ROWS, CHANNELS_PAD]
    partial_ptr,     # f32  [4, NUM_TILES, CHANNELS_PAD]
    ROWS_TOTAL: ct.Constant[int],
    CHANNELS_ACTUAL: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    NUM_TILES_: ct.Constant[int],
):
    tile_r = ct.bid(0)
    tile_c = ct.bid(1)

    # Row and col index tiles (only col mask matters for validity; rows are pow2-aligned).
    row_idx = tile_r * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    col_idx = tile_c * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    row_mask = row_idx < ROWS_TOTAL
    col_mask = col_idx < CHANNELS_ACTUAL
    mask_2d = ct.reshape(row_mask, (BLOCK_R, 1)) & ct.reshape(col_mask, (1, BLOCK_C))

    zero_2d = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)

    # Load tiles (padded).
    x = ct.astype(
        ct.load(x_ptr, index=(tile_r, tile_c), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    norm = ct.load(norm_ptr, index=(tile_r, tile_c), shape=(BLOCK_R, BLOCK_C))
    residual = ct.load(residual_ptr, index=(tile_r, tile_c),
                       shape=(BLOCK_R, BLOCK_C))
    side_in = ct.astype(
        ct.load(side_in_ptr, index=(tile_r, tile_c), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )

    gamma_1d = ct.load(gamma_ptr, index=(tile_c,), shape=(BLOCK_C,))
    gamma_2d = ct.reshape(gamma_1d, (1, BLOCK_C))
    side_scale_1d = ct.load(side_scale_ptr, index=(tile_c,), shape=(BLOCK_C,))
    side_scale_2d = ct.reshape(side_scale_1d, (1, BLOCK_C))

    row_sum = ct.load(row_sum_ptr, index=(tile_r,), shape=(BLOCK_R,))
    row_dot = ct.load(row_dot_ptr, index=(tile_r,), shape=(BLOCK_R,))
    row_sum_2d = ct.reshape(row_sum, (BLOCK_R, 1))
    row_dot_2d = ct.reshape(row_dot, (BLOCK_R, 1))

    # Scale is preflattened to [ROWS] contiguous (matches per-row gather).
    scale_row = ct.load(scale_ptr, index=(tile_r,), shape=(BLOCK_R,))
    scale_row = ct.astype(scale_row, ct.float32)
    scale_row_2d = ct.reshape(scale_row, (BLOCK_R, 1))

    weighted = x * gamma_2d
    centered = weighted * float(CHANNELS_ACTUAL) - row_sum_2d - norm * row_dot_2d
    grad = scale_row_2d * centered
    add_val = residual + grad
    side_val_f = add_val * side_scale_2d
    # Round side_val to bf16 then back to f32 (matches Triton's inline PTX rounding).
    side_val_bf = ct.astype(side_val_f, ct.bfloat16)
    side_val_f_rn = ct.astype(side_val_bf, ct.float32)

    ct.store(add_out_ptr, index=(tile_r, tile_c), tile=add_val)
    ct.store(side_out_ptr, index=(tile_r, tile_c), tile=side_val_bf)

    # Column partials (reduce axis=0).
    out0_m = ct.where(mask_2d, x * norm, zero_2d)
    out1_m = ct.where(mask_2d, x, zero_2d)
    out3_m = ct.where(mask_2d, add_val * side_in, zero_2d)
    out6_m = ct.where(mask_2d, side_val_f_rn, zero_2d)

    out0 = ct.sum(out0_m, axis=0)  # (BLOCK_C,)
    out1 = ct.sum(out1_m, axis=0)
    out3 = ct.sum(out3_m, axis=0)
    out6 = ct.sum(out6_m, axis=0)

    # partial is [4, NUM_TILES, CHANNELS_PAD]. Store using tile-space index (plane, tile_r, tile_c).
    ct.store(partial_ptr, index=(0, tile_r, tile_c),
             tile=ct.reshape(out0, (1, 1, BLOCK_C)))
    ct.store(partial_ptr, index=(1, tile_r, tile_c),
             tile=ct.reshape(out1, (1, 1, BLOCK_C)))
    ct.store(partial_ptr, index=(2, tile_r, tile_c),
             tile=ct.reshape(out3, (1, 1, BLOCK_C)))
    ct.store(partial_ptr, index=(3, tile_r, tile_c),
             tile=ct.reshape(out6, (1, 1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_ptr,     # f32 [4, NUM_TILES, CHANNELS_PAD]
    out0_ptr,        # f32 [CHANNELS_PAD]
    out1_ptr,        # f32 [CHANNELS_PAD]
    out3_ptr,        # f32 [CHANNELS_PAD]
    out6_ptr,        # f32 [CHANNELS_PAD]
    NUM_TILES_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_T: ct.Constant[int],  # power-of-2 >= NUM_TILES
):
    col_block = ct.bid(0)
    # Load (BLOCK_T, BLOCK_C) tiles from each plane, using tile-space index (plane, 0, col_block).
    p0 = ct.load(partial_ptr, index=(0, 0, col_block),
                 shape=(1, BLOCK_T, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    p1 = ct.load(partial_ptr, index=(1, 0, col_block),
                 shape=(1, BLOCK_T, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    p3 = ct.load(partial_ptr, index=(2, 0, col_block),
                 shape=(1, BLOCK_T, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    p6 = ct.load(partial_ptr, index=(3, 0, col_block),
                 shape=(1, BLOCK_T, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    p0_2d = ct.reshape(p0, (BLOCK_T, BLOCK_C))
    p1_2d = ct.reshape(p1, (BLOCK_T, BLOCK_C))
    p3_2d = ct.reshape(p3, (BLOCK_T, BLOCK_C))
    p6_2d = ct.reshape(p6, (BLOCK_T, BLOCK_C))

    tile_idx = ct.arange(BLOCK_T, dtype=ct.int32)
    valid_1d = tile_idx < NUM_TILES_
    valid_2d = ct.reshape(valid_1d, (BLOCK_T, 1))
    zero_2d = ct.zeros((BLOCK_T, BLOCK_C), dtype=ct.float32)

    s0 = ct.sum(ct.where(valid_2d, p0_2d, zero_2d), axis=0)
    s1 = ct.sum(ct.where(valid_2d, p1_2d, zero_2d), axis=0)
    s3 = ct.sum(ct.where(valid_2d, p3_2d, zero_2d), axis=0)
    s6 = ct.sum(ct.where(valid_2d, p6_2d, zero_2d), axis=0)
    s6_bf = ct.astype(s6, ct.bfloat16)
    s6_out = ct.astype(s6_bf, ct.float32)

    ct.store(out0_ptr, index=(col_block,), tile=s0)
    ct.store(out1_ptr, index=(col_block,), tile=s1)
    ct.store(out3_ptr, index=(col_block,), tile=s3)
    ct.store(out6_ptr, index=(col_block,), tile=s6_out)


def _ceil_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="df4617bb")
def oracle_forward(inputs):
    x, gamma, norm, scale, residual, side_in, side_scale, *_shape_params = inputs
    device = x.device

    # Pad channel dim of the (ROWS, CHANNELS) tensors to power-of-2 (=ROW_C_BLOCK).
    C_PAD = ROW_C_BLOCK
    x_pad = torch.zeros((ROWS, C_PAD), device=device, dtype=x.dtype)
    x_pad[:, :CHANNELS].copy_(x.view(ROWS, CHANNELS))
    norm_pad = torch.zeros((ROWS, C_PAD), device=device, dtype=torch.float32)
    norm_pad[:, :CHANNELS].copy_(norm.view(ROWS, CHANNELS))
    residual_pad = torch.zeros((ROWS, C_PAD), device=device, dtype=torch.float32)
    residual_pad[:, :CHANNELS].copy_(residual.view(ROWS, CHANNELS))
    side_in_pad = torch.zeros((ROWS, C_PAD), device=device, dtype=side_in.dtype)
    side_in_pad[:, :CHANNELS].copy_(side_in.view(ROWS, CHANNELS))
    gamma_pad = torch.zeros((C_PAD,), device=device, dtype=torch.float32)
    gamma_pad[:CHANNELS].copy_(gamma)
    side_scale_pad = torch.zeros((C_PAD,), device=device, dtype=torch.float32)
    side_scale_pad[:CHANNELS].copy_(side_scale)

    # Flatten scale [128, 1370, 1] with stride (1376, 1, 1) to a dense [ROWS] tensor.
    scale_flat = scale.reshape(ROWS).contiguous()

    # Outputs.
    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    add_out_pad = torch.empty((ROWS, C_PAD), device=device, dtype=torch.float32)
    side_out_pad = torch.empty((ROWS, C_PAD), device=device, dtype=torch.bfloat16)
    partial = torch.zeros((4, NUM_TILES, C_PAD), device=device, dtype=torch.float32)
    out0_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    out1_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    out3_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    out6_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    # Kernel 1: row reductions.
    ct.launch(
        stream,
        (_cdiv(ROWS, ROW_BLOCK), 1, 1),
        _row_reductions_kernel,
        (x_pad, gamma_pad, norm_pad, row_sum, row_dot,
         CHANNELS, ROW_BLOCK, C_PAD),
    )

    # Kernel 2: epilogue with side outputs + column partials.
    # Grid: (NUM_TILES, C_PAD // TILE_COLS)
    if C_PAD % TILE_COLS != 0:
        raise NotImplementedError("TILE_COLS must divide C_PAD")
    ct.launch(
        stream,
        (NUM_TILES, C_PAD // TILE_COLS, 1),
        _epilogue_reduce_kernel,
        (x_pad, gamma_pad, norm_pad, scale_flat, residual_pad, side_scale_pad,
         side_in_pad, row_sum, row_dot, add_out_pad, side_out_pad, partial,
         ROWS, CHANNELS, TILE_ROWS, TILE_COLS, NUM_TILES),
    )

    # Kernel 3: finalize column reductions.
    tiles_pad = _ceil_pow2(NUM_TILES)
    ct.launch(
        stream,
        (C_PAD // TILE_COLS, 1, 1),
        _finalize_kernel,
        (partial, out0_pad, out1_pad, out3_pad, out6_pad,
         NUM_TILES, TILE_COLS, tiles_pad),
    )

    # Narrow padded outputs back to CHANNELS=768.
    add_out = torch.empty_strided(ADD_SHAPE, ADD_STRIDE, device=device, dtype=torch.float32)
    add_out.view(ROWS, CHANNELS).copy_(add_out_pad[:, :CHANNELS])
    side_out = torch.empty_strided(SIDE_SHAPE, SIDE_STRIDE, device=device, dtype=torch.bfloat16)
    side_out.copy_(side_out_pad[:, :CHANNELS])
    out0 = out0_pad[:CHANNELS].contiguous()
    out1 = out1_pad[:CHANNELS].contiguous()
    out3 = out3_pad[:CHANNELS].contiguous()
    out6 = out6_pad[:CHANNELS].contiguous()

    return out0, out1, add_out, out3, side_out, side_out.permute(1, 0), out6
