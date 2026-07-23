"""cuTile port of sum_sum_sum_74daaa264688: GPT-J LayerNorm backward.

Fair port: 2 kernels matching Triton structure.
Kernel 1 (`_store_row_partials`): row-tiled LN backward. Writes the returned
add_out (f32) and bf16_out. Accumulates partial column sums for x*normed, x,
and bf16(side).float().
Kernel 2 (`_finalize_partials`): finalizes the 3 partial column sums with a
bf16 rounding boundary for the side sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
CHANNELS = 4096
ROW_FACTOR = 4096.0
INV_ROW_FACTOR = 1.0 / 4096.0


@ct.kernel
def _store_row_partials_kernel(
    arg0_ptr, arg1_ptr, arg2_ptr, arg3_ptr,  # bf16 (ROWS, CHANNELS)
    weight_ptr,      # f32 (CHANNELS,)
    norm_bf16_ptr,   # bf16 (ROWS, CHANNELS)
    norm_f32_ptr,    # f32 (ROWS, CHANNELS)
    mean_ptr,        # f32 (ROWS,)
    scale_ptr,       # f32 (ROWS,)
    residual_ptr,    # f32 (ROWS, CHANNELS)
    add_out_ptr,     # f32 (ROWS, CHANNELS)
    bf16_out_ptr,    # bf16 (ROWS, CHANNELS)
    partials_ptr,    # f32 (ROWS_, 3, CHANNELS)  (one per row group; ROWS_PER_GROUP=1)
    CHANNELS_: ct.Constant[int],
):
    row = ct.bid(0)
    weight = ct.load(weight_ptr, index=(0,), shape=(CHANNELS_,))
    weight_2d = ct.reshape(weight, (1, CHANNELS_))

    x = ct.astype(ct.load(arg0_ptr, index=(row, 0), shape=(1, CHANNELS_)), ct.float32)
    x = x + ct.astype(ct.load(arg1_ptr, index=(row, 0), shape=(1, CHANNELS_)), ct.float32)
    x = x + ct.astype(ct.load(arg2_ptr, index=(row, 0), shape=(1, CHANNELS_)), ct.float32)
    x = x + ct.astype(ct.load(arg3_ptr, index=(row, 0), shape=(1, CHANNELS_)), ct.float32)

    norm_src = ct.astype(ct.load(norm_bf16_ptr, index=(row, 0), shape=(1, CHANNELS_)), ct.float32)
    norm_src = norm_src + ct.load(norm_f32_ptr, index=(row, 0), shape=(1, CHANNELS_))
    mean_tile = ct.reshape(ct.load(mean_ptr, index=(row,), shape=(1,)), (1, 1))
    scale_tile = ct.reshape(ct.load(scale_ptr, index=(row,), shape=(1,)), (1, 1))
    normed = (norm_src - mean_tile) * scale_tile

    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, CHANNELS_))
    weighted = x * weight_2d
    row_sum = ct.sum(weighted, axis=1, keepdims=True)
    weighted_norm = weighted * normed
    row_dot = ct.sum(weighted_norm, axis=1, keepdims=True)

    centered = weighted * ROW_FACTOR - row_sum
    centered = centered - normed * row_dot
    grad = scale_tile * INV_ROW_FACTOR * centered
    side = residual + grad
    side_bf16 = ct.astype(side, ct.bfloat16)

    ct.store(add_out_ptr, index=(row, 0), tile=side)
    ct.store(bf16_out_ptr, index=(row, 0), tile=side_bf16)

    # Partial column sums for this single-row group.
    acc_x_norm = ct.reshape(x * normed, (1, 1, CHANNELS_))
    acc_x = ct.reshape(x, (1, 1, CHANNELS_))
    acc_bf16_side = ct.reshape(ct.astype(side_bf16, ct.float32), (1, 1, CHANNELS_))
    ct.store(partials_ptr, index=(row, 0, 0), tile=acc_x_norm)
    ct.store(partials_ptr, index=(row, 1, 0), tile=acc_x)
    ct.store(partials_ptr, index=(row, 2, 0), tile=acc_bf16_side)


@ct.kernel
def _finalize_partials_kernel(
    partials_ptr,       # f32 (NUM_GROUPS, 3, CHANNELS)
    out_x_norm_ptr,     # f32 (CHANNELS,)
    out_x_ptr,          # f32 (CHANNELS,)
    out_bf16_sum_ptr,   # f32 (CHANNELS,)
    NUM_GROUPS: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)

    x_norm = ct.load(partials_ptr, index=(0, 0, col_block), shape=(NUM_GROUPS, 1, FINAL_BLOCK_C))
    x = ct.load(partials_ptr, index=(0, 1, col_block), shape=(NUM_GROUPS, 1, FINAL_BLOCK_C))
    bf16_side = ct.load(partials_ptr, index=(0, 2, col_block), shape=(NUM_GROUPS, 1, FINAL_BLOCK_C))

    s_x_norm = ct.reshape(ct.sum(x_norm, axis=0), (FINAL_BLOCK_C,))
    s_x = ct.reshape(ct.sum(x, axis=0), (FINAL_BLOCK_C,))
    s_bf16 = ct.reshape(ct.sum(bf16_side, axis=0), (FINAL_BLOCK_C,))
    side_sum = ct.astype(ct.astype(s_bf16, ct.bfloat16), ct.float32)

    ct.store(out_x_norm_ptr, index=(col_block,), tile=s_x_norm)
    ct.store(out_x_ptr, index=(col_block,), tile=s_x)
    ct.store(out_bf16_sum_ptr, index=(col_block,), tile=side_sum)


@oracle_impl(hardware="B200", point="85287631", FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, FINAL_BLOCK_C: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1,
        weight, norm_bf16, norm_f32, mean, scale, residual,
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    add_out = torch.empty_strided(
        (1, ROWS, CHANNELS),
        (ROWS * CHANNELS, CHANNELS, 1),
        device=device, dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        (ROWS, CHANNELS),
        (CHANNELS, 1),
        device=device, dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (ROWS, 3, CHANNELS), device=device, dtype=torch.float32
    )
    out_x_norm = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_bf16_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    norm_bf16_2d = norm_bf16.view(ROWS, CHANNELS)
    norm_f32_2d = norm_f32.view(ROWS, CHANNELS)
    mean_1d = mean.view(ROWS)
    scale_1d = scale.view(ROWS)
    residual_2d = residual.view(ROWS, CHANNELS)
    add_out_2d = add_out.view(ROWS, CHANNELS)

    ct.launch(
        stream,
        (ROWS, 1, 1),
        _store_row_partials_kernel,
        (
            arg0_1, arg1_1, arg2_1, arg3_1,
            weight, norm_bf16_2d, norm_f32_2d, mean_1d, scale_1d, residual_2d,
            add_out_2d, bf16_out, partials,
            CHANNELS,
        ),
    )
    ct.launch(
        stream,
        (CHANNELS // FINAL_BLOCK_C, 1, 1),
        _finalize_partials_kernel,
        (partials, out_x_norm, out_x, out_bf16_sum, ROWS, FINAL_BLOCK_C),
    )
    return out_x_norm, out_x, add_out, bf16_out, bf16_out.permute(1, 0), out_bf16_sum
