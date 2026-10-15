"""cuTile port of sum_sum_sum_23b40b0be2a8: BEiT LN-backward tail. Two kernels:
  1) Row-tile kernel: per row-tile compute grad, store bf16 patch output for
     non-CLS tokens, and per-tile column partials for four reductions.
  2) Finalize kernel: sum partials over tiles for each channel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
PATCH_START = 1
PATCH_TOKENS = 196
PATCH_HW = 14
CHANNELS = 768
ROWS = BATCH * TOKENS
INV_CHANNELS = 1.0 / CHANNELS


@ct.kernel
def _row_tile_kernel(
    x_ptr,             # bf16 [ROWS, CHANNELS]
    weight_ptr,        # f32 [CHANNELS]
    source_ptr,        # f32 [ROWS, CHANNELS]
    mean_ptr,          # f32 [ROWS]
    scale_ptr,         # f32 [ROWS]
    residual_ptr,      # f32 [ROWS, CHANNELS]
    partial_x_centered_ptr,  # f32 [NUM_TILES, CHANNELS]
    partial_x_ptr,           # f32 [NUM_TILES, CHANNELS]
    partial_cls_ptr,         # f32 [NUM_TILES, CHANNELS]
    partial_patch_ptr,       # f32 [NUM_TILES, CHANNELS]
    patch_out_ptr,     # bf16 flat [BATCH * CHANNELS * PATCH_HW * PATCH_HW]
    ROWS_: ct.Constant[int],
    TOKENS_: ct.Constant[int],
    PATCH_START_: ct.Constant[int],
    PATCH_TOKENS_: ct.Constant[int],
    PATCH_HW_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    INV_CHANNELS_: ct.Constant[float],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_CHANNELS: ct.Constant[int],
):
    tile_row = ct.bid(0)
    rows_1d = tile_row * BLOCK_ROWS + ct.arange(BLOCK_ROWS, dtype=ct.int32)
    channels_1d = ct.arange(BLOCK_CHANNELS, dtype=ct.int32)
    row_mask_1d = rows_1d < ROWS_
    channel_mask_1d = channels_1d < CHANNELS_
    row_mask_2d = ct.reshape(row_mask_1d, (BLOCK_ROWS, 1))
    channel_mask_2d = ct.reshape(channel_mask_1d, (1, BLOCK_CHANNELS))
    zero_bool_2d = ct.full((BLOCK_ROWS, BLOCK_CHANNELS), False, dtype=ct.bool_)
    mask = (row_mask_2d | zero_bool_2d) & (channel_mask_2d | zero_bool_2d)

    token_1d = rows_1d - (rows_1d // TOKENS_) * TOKENS_
    batch_1d = rows_1d // TOKENS_
    token_2d = ct.reshape(token_1d, (BLOCK_ROWS, 1))
    batch_2d = ct.reshape(batch_1d, (BLOCK_ROWS, 1))
    zero_i32_2d = ct.zeros((BLOCK_ROWS, BLOCK_CHANNELS), dtype=ct.int32)
    channels_2d = ct.reshape(channels_1d, (1, BLOCK_CHANNELS)) + zero_i32_2d
    rows_2d = ct.reshape(rows_1d, (BLOCK_ROWS, 1)) + zero_i32_2d

    offsets_2d = rows_2d * CHANNELS_ + channels_2d

    zero_f_2d = ct.zeros((BLOCK_ROWS, BLOCK_CHANNELS), dtype=ct.float32)

    # x is bf16 [ROWS, CHANNELS] - load via gather with 2D indices
    x = ct.astype(
        ct.gather(x_ptr, (offsets_2d,), mask=mask, padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    source = ct.gather(source_ptr, (offsets_2d,), mask=mask, padding_value=ct.float32(0.0))
    residual = ct.gather(residual_ptr, (offsets_2d,), mask=mask, padding_value=ct.float32(0.0))

    # weight is 1D
    weight_1d = ct.gather(weight_ptr, (channels_1d,), mask=channel_mask_1d, padding_value=ct.float32(0.0))
    mean_1d = ct.gather(mean_ptr, (rows_1d,), mask=row_mask_1d, padding_value=ct.float32(0.0))
    scale_1d = ct.gather(scale_ptr, (rows_1d,), mask=row_mask_1d, padding_value=ct.float32(0.0))

    # broadcast
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_CHANNELS))
    mean_row = ct.reshape(mean_1d, (BLOCK_ROWS, 1))
    scale_row = ct.reshape(scale_1d, (BLOCK_ROWS, 1))

    centered = (source - mean_row) * scale_row
    weighted = x * weight_2d
    row_sum = ct.sum(ct.where(mask, weighted, zero_f_2d), axis=1)
    row_dot = ct.sum(ct.where(mask, weighted * centered, zero_f_2d), axis=1)
    row_sum_2d = ct.reshape(row_sum, (BLOCK_ROWS, 1))
    row_dot_2d = ct.reshape(row_dot, (BLOCK_ROWS, 1))

    term = weighted * float(CHANNELS_) - row_sum_2d
    term = term - centered * row_dot_2d
    grad = (scale_row * INV_CHANNELS_) * term
    add_value = residual + grad

    # cls_mask = mask & (token == 0), patch_mask = mask & (token >= PATCH_START)
    is_cls_2d = (token_2d == 0) | zero_bool_2d
    is_patch_2d = (token_2d >= PATCH_START_) | zero_bool_2d
    cls_mask = mask & is_cls_2d
    patch_mask = mask & is_patch_2d
    patch_index_2d = token_2d - PATCH_START_

    patch_offsets = batch_2d * (PATCH_TOKENS_ * CHANNELS_) + patch_index_2d * CHANNELS_ + channels_2d
    patch_bf16 = ct.astype(add_value, ct.bfloat16)
    ct.scatter(patch_out_ptr, (patch_offsets,), patch_bf16, mask=patch_mask)

    # partial sums: sum along rows (axis=0) → (BLOCK_CHANNELS,)
    partial_x_centered_val = ct.sum(ct.where(mask, x * centered, zero_f_2d), axis=0)
    partial_x_val = ct.sum(ct.where(mask, x, zero_f_2d), axis=0)
    partial_cls_val = ct.sum(ct.where(cls_mask, add_value, zero_f_2d), axis=0)
    partial_patch_val = ct.sum(
        ct.where(patch_mask, ct.astype(patch_bf16, ct.float32), zero_f_2d),
        axis=0,
    )

    partial_offsets = tile_row * CHANNELS_ + channels_1d
    ct.scatter(partial_x_centered_ptr, (partial_offsets,), partial_x_centered_val, mask=channel_mask_1d)
    ct.scatter(partial_x_ptr, (partial_offsets,), partial_x_val, mask=channel_mask_1d)
    ct.scatter(partial_cls_ptr, (partial_offsets,), partial_cls_val, mask=channel_mask_1d)
    ct.scatter(partial_patch_ptr, (partial_offsets,), partial_patch_val, mask=channel_mask_1d)


@ct.kernel
def _finalize_kernel(
    partial_x_centered_ptr,  # f32 [NUM_TILES * CHANNELS]
    partial_x_ptr,
    partial_cls_ptr,
    partial_patch_ptr,
    out_x_centered_ptr,      # f32 [CHANNELS]
    out_x_ptr,               # f32 [CHANNELS]
    out_cls_ptr,             # f32 [1, 1, CHANNELS]
    out_patch_sum_ptr,       # f32 [CHANNELS]
    NUM_ROW_TILES: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_CHANNELS: ct.Constant[int],
):
    c_block = ct.bid(0)
    channels_1d = c_block * BLOCK_CHANNELS + ct.arange(BLOCK_CHANNELS, dtype=ct.int32)
    channel_mask = channels_1d < CHANNELS_

    # BLOCK_TILES rounded up to cover NUM_ROW_TILES. But NUM_ROW_TILES may be > BLOCK_TILES.
    # For simplicity, loop over tile chunks.
    acc_x_centered = ct.zeros((BLOCK_CHANNELS,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_CHANNELS,), dtype=ct.float32)
    acc_cls = ct.zeros((BLOCK_CHANNELS,), dtype=ct.float32)
    acc_patch = ct.zeros((BLOCK_CHANNELS,), dtype=ct.float32)

    # Build indices for (BLOCK_TILES, BLOCK_CHANNELS)
    zero_i32_2d = ct.zeros((BLOCK_TILES, BLOCK_CHANNELS), dtype=ct.int32)
    zero_bool_2d = ct.full((BLOCK_TILES, BLOCK_CHANNELS), False, dtype=ct.bool_)
    channel_mask_2d = ct.reshape(channel_mask, (1, BLOCK_CHANNELS)) | zero_bool_2d
    channels_2d = ct.reshape(channels_1d, (1, BLOCK_CHANNELS)) + zero_i32_2d
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_CHANNELS), dtype=ct.float32)

    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tile_1d = tile_base + ct.arange(BLOCK_TILES, dtype=ct.int32)
        tile_mask_1d = tile_1d < NUM_ROW_TILES
        tile_mask_2d = ct.reshape(tile_mask_1d, (BLOCK_TILES, 1)) | zero_bool_2d
        mask = tile_mask_2d & channel_mask_2d
        tiles_2d = ct.reshape(tile_1d, (BLOCK_TILES, 1)) + zero_i32_2d
        offsets = tiles_2d * CHANNELS_ + channels_2d

        acc_x_centered = acc_x_centered + ct.sum(
            ct.gather(partial_x_centered_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0)),
            axis=0,
        )
        acc_x = acc_x + ct.sum(
            ct.gather(partial_x_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0)),
            axis=0,
        )
        acc_cls = acc_cls + ct.sum(
            ct.gather(partial_cls_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0)),
            axis=0,
        )
        acc_patch = acc_patch + ct.sum(
            ct.gather(partial_patch_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0)),
            axis=0,
        )

    ct.scatter(out_x_centered_ptr, (channels_1d,), acc_x_centered, mask=channel_mask)
    ct.scatter(out_x_ptr, (channels_1d,), acc_x, mask=channel_mask)
    ct.scatter(out_cls_ptr, (channels_1d,), acc_cls, mask=channel_mask)
    acc_patch_bf16 = ct.astype(ct.astype(acc_patch, ct.bfloat16), ct.float32)
    ct.scatter(out_patch_sum_ptr, (channels_1d,), acc_patch_bf16, mask=channel_mask)


@oracle_impl(
    hardware="B200",
    point="49c36e1b",
    TILE_ROWS=16,
    TILE_CHANNELS=1024,
    FINAL_BLOCK_CHANNELS=16,
    FINAL_BLOCK_TILES=1024,
)
def oracle_forward(
    inputs,
    *,
    TILE_ROWS: int,
    TILE_CHANNELS: int,
    FINAL_BLOCK_CHANNELS: int,
    FINAL_BLOCK_TILES: int,
):
    (
        x,
        weight,
        source,
        mean,
        scale,
        residual,
        _shape0,
        _shape1,
    ) = inputs

    device = x.device
    num_row_tiles = (ROWS + TILE_ROWS - 1) // TILE_ROWS
    partials = torch.empty((4, num_row_tiles, CHANNELS), device=device, dtype=torch.float32)
    out_x_centered = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_cls = torch.empty((1, 1, CHANNELS), device=device, dtype=torch.float32)
    # patch_out is channels-last strided: (BATCH, CHANNELS, PATCH_HW, PATCH_HW) with
    # stride (PATCH_TOKENS * CHANNELS, 1, PATCH_HW * CHANNELS, CHANNELS).
    # This is NHWC memory. Use NHWC contiguous alloc then permute.
    patch_out_nhwc = torch.empty(
        (BATCH, PATCH_HW, PATCH_HW, CHANNELS),
        device=device,
        dtype=torch.bfloat16,
    )
    patch_out = patch_out_nhwc.permute(0, 3, 1, 2)  # (BATCH, CHANNELS, PATCH_HW, PATCH_HW)
    out_patch_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    # source, mean, scale, residual are 3D [BATCH, TOKENS, C or 1]. Flatten.
    source_2d = source.view(ROWS, CHANNELS)
    residual_2d = residual.view(ROWS, CHANNELS)
    mean_1d = mean.view(ROWS)
    scale_1d = scale.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_tiles, 1, 1),
        _row_tile_kernel,
        (
            x.view(-1), weight, source_2d.reshape(-1), mean_1d, scale_1d, residual_2d.reshape(-1),
            partials[0].view(-1), partials[1].view(-1),
            partials[2].view(-1), partials[3].view(-1),
            patch_out_nhwc.view(-1),
            ROWS, TOKENS, PATCH_START, PATCH_TOKENS, PATCH_HW, CHANNELS, INV_CHANNELS,
            TILE_ROWS, TILE_CHANNELS,
        ),
    )

    ct.launch(
        stream,
        ((CHANNELS + FINAL_BLOCK_CHANNELS - 1) // FINAL_BLOCK_CHANNELS, 1, 1),
        _finalize_kernel,
        (
            partials[0].view(-1), partials[1].view(-1),
            partials[2].view(-1), partials[3].view(-1),
            out_x_centered, out_x, out_cls.view(-1), out_patch_sum,
            num_row_tiles, CHANNELS, FINAL_BLOCK_TILES, FINAL_BLOCK_CHANNELS,
        ),
    )

    return out_x_centered, out_x, out_cls, patch_out, out_patch_sum
