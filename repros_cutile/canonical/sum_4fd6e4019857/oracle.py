"""cuTile port of sum_4fd6e4019857: wide-row bf16 softmax-backward.

Two kernels, matching Triton's structure:
  1. _partial_sum_kernel: for each (row, tile), compute rounded prob * grad
     and store the per-tile partial (in-kernel `ct.sum`).
  2. _softmax_backward_epilogue_kernel: read all partials for a row,
     reduce them (in-kernel `ct.sum`), recompute prob, and store the
     FMA epilogue as bf16.

No host-side `partials.sum(dim=1)` — the tile-partials reduction is done
inside the epilogue kernel just like Triton does.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 262144


@ct.kernel
def _partial_sum_kernel(
    x_ptr,          # bf16 [ROWS, COLS]
    row_max_ptr,    # f32  [ROWS]
    row_denom_ptr,  # f32  [ROWS]
    grad_ptr,       # bf16 [1]
    partial_ptr,    # f32  [ROWS, NUM_TILES]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    tile = ct.bid(1)
    x = ct.load(x_ptr, index=(row, tile), shape=(1, BLOCK_N))
    xf = ct.astype(x, ct.float32)

    row_max = ct.load(row_max_ptr, index=(row,), shape=(1,))
    row_denom = ct.load(row_denom_ptr, index=(row,), shape=(1,))
    grad = ct.load(grad_ptr, index=(0,), shape=(1,))
    grad_f = ct.astype(grad, ct.float32)

    row_max_2d = ct.reshape(row_max, (1, 1))
    row_denom_2d = ct.reshape(row_denom, (1, 1))
    prob_f32 = ct.exp(xf - row_max_2d) / row_denom_2d
    prob_bf = ct.astype(prob_f32, ct.bfloat16)
    prob = ct.astype(prob_bf, ct.float32)

    grad_2d = ct.reshape(grad_f, (1, 1))
    prod = prob * grad_2d
    partial_val = ct.sum(prod)
    partial_tile = ct.full((1, 1), 0.0, dtype=ct.float32) + partial_val
    ct.store(partial_ptr, index=(row, tile), tile=partial_tile)


@ct.kernel
def _softmax_backward_epilogue_kernel(
    x_ptr,          # bf16 [ROWS, COLS]
    row_max_ptr,    # f32  [ROWS]
    row_denom_ptr,  # f32  [ROWS]
    grad_ptr,       # bf16 [1]
    partial_ptr,    # f32  [ROWS, NUM_TILES]
    out_ptr,        # bf16 [ROWS, COLS]
    NUM_TILES_C: ct.Constant[int],
    PARTIAL_BLOCK_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    tile = ct.bid(1)

    # In-kernel reduce over the row's partial tiles.
    partials = ct.load(
        partial_ptr, index=(row, 0), shape=(1, PARTIAL_BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    idx = ct.arange(PARTIAL_BLOCK_C, dtype=ct.int32)
    valid = ct.reshape(idx < NUM_TILES_C, (1, PARTIAL_BLOCK_C))
    zero = ct.zeros((1, PARTIAL_BLOCK_C), dtype=ct.float32)
    partials_valid = ct.where(valid, partials, zero)
    product_sum = ct.sum(partials_valid)  # scalar
    product_sum_2d = ct.full((1, 1), 0.0, dtype=ct.float32) + product_sum

    # Recompute prob for this output tile.
    x = ct.load(x_ptr, index=(row, tile), shape=(1, BLOCK_N))
    xf = ct.astype(x, ct.float32)
    row_max = ct.load(row_max_ptr, index=(row,), shape=(1,))
    row_denom = ct.load(row_denom_ptr, index=(row,), shape=(1,))
    grad = ct.load(grad_ptr, index=(0,), shape=(1,))
    grad_f = ct.astype(grad, ct.float32)

    row_max_2d = ct.reshape(row_max, (1, 1))
    row_denom_2d = ct.reshape(row_denom, (1, 1))
    grad_2d = ct.reshape(grad_f, (1, 1))

    prob_f32 = ct.exp(xf - row_max_2d) / row_denom_2d
    prob_bf = ct.astype(prob_f32, ct.bfloat16)
    prob = ct.astype(prob_bf, ct.float32)
    product = grad_2d * prob
    out = product - prob * product_sum_2d
    ct.store(out_ptr, index=(row, tile), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="H100", point="74d25999", PARTIAL_BLOCK_N=16384, OUTPUT_BLOCK_N=1024)
@oracle_impl(hardware="B200", point="74d25999", PARTIAL_BLOCK_N=16384, OUTPUT_BLOCK_N=1024)
def oracle_forward(inputs, *, PARTIAL_BLOCK_N: int, OUTPUT_BLOCK_N: int):
    x, row_max, row_denom, grad_scalar, shape = inputs
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    out_shape = tuple(int(dim) for dim in shape)

    row_max_flat = row_max.view(rows)
    row_denom_flat = row_denom.view(rows)
    grad_flat = grad_scalar.view(1)

    partial_tiles = cols // PARTIAL_BLOCK_N
    partial_block = 1 << (partial_tiles - 1).bit_length() if partial_tiles > 1 else 1
    partials = torch.empty((rows, partial_tiles), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, partial_tiles, 1),
        _partial_sum_kernel,
        (x, row_max_flat, row_denom_flat, grad_flat, partials, PARTIAL_BLOCK_N),
    )

    out = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
    output_tiles = cols // OUTPUT_BLOCK_N
    ct.launch(
        stream,
        (rows, output_tiles, 1),
        _softmax_backward_epilogue_kernel,
        (x, row_max_flat, row_denom_flat, grad_flat, partials, out,
         partial_tiles, partial_block, OUTPUT_BLOCK_N),
    )
    return out
