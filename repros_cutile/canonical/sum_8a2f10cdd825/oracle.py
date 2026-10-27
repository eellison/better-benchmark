"""cuTile port of sum_8a2f10cdd825: MobileBERT bf16 masked materialization + column sum."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
COLS = 512


@ct.kernel
def _masked_store_partial_kernel(
    x_ptr,       # (ROWS, COLS) bf16
    mask_ptr,    # (ROWS, COLS) b8
    fill_ptr,    # bf16[1] (scalar)
    out_ptr,     # (ROWS, COLS) bf16
    partial_ptr, # (num_row_tiles, COLS) f32
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)

    m = ct.load(mask_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N))
    x = ct.load(x_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar = ct.astype(fill, ct.float32)
    # Note: fill is 1D shape(1,), need to make it broadcastable to (BLOCK_M, BLOCK_N)
    # cuTile broadcasting: convert to fp32 arithmetic
    x_f = ct.astype(x, ct.float32)
    fill_f = ct.reshape(fill_scalar, (1, 1))
    values = ct.where(m, fill_f, x_f)
    values_bf16 = ct.astype(values, ct.bfloat16)
    ct.store(out_ptr, index=(row_tile, col_tile), tile=values_bf16)

    # Column sum on this tile, then store as partial row
    col_sum = ct.sum(values, axis=0, keepdims=True)  # (1, BLOCK_N)
    ct.store(partial_ptr, index=(row_tile, col_tile), tile=col_sum)


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr,  # (NUM_ROW_TILES, COLS) f32
    sum_ptr,      # (COLS,) f32
    NUM_ROW_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)
    partials = ct.load(partial_ptr, index=(0, col_tile), shape=(NUM_ROW_TILES, BLOCK_N))
    sums = ct.sum(partials, axis=0)
    # bf16 round-trip: cast to bf16 then back to f32
    sums_bf16 = ct.astype(sums, ct.bfloat16)
    sums_f32 = ct.astype(sums_bf16, ct.float32)
    ct.store(sum_ptr, index=(col_tile,), tile=sums_f32)


@oracle_impl(hardware="B200", point="c5bb71a2", BLOCK_M=128, BLOCK_N=64, FINAL_BLOCK_N=32)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, FINAL_BLOCK_N):
    x, mask, fill, _shape0, _shape1, _shape2 = inputs

    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((COLS,), device=x.device, dtype=torch.float32)
    num_row_tiles = (ROWS + BLOCK_M - 1) // BLOCK_M
    partial = torch.empty((num_row_tiles, COLS), device=x.device, dtype=torch.float32)

    # Mask is [256, 128, 512] but semantically [32768, 512] after flat view.
    mask_2d = mask.view(ROWS, COLS)
    fill_v = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_tiles, COLS // BLOCK_N, 1),
        _masked_store_partial_kernel,
        (x, mask_2d, fill_v, out, partial, BLOCK_M, BLOCK_N),
    )
    ct.launch(
        stream,
        (COLS // FINAL_BLOCK_N, 1, 1),
        _finalize_sum_kernel,
        (partial, sum_out, num_row_tiles, FINAL_BLOCK_N),
    )
    return out, out.permute(1, 0), sum_out
