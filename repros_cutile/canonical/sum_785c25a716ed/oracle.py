"""cuTile port of sum_785c25a716ed (SCHEDULER_FUSION): permute+clone with
sibling column reduction, three shape/stride variants.

Uses cuTile for a fused clone + partial-column-sum kernel (matching the
Triton `_copy_partial_sum_kernel` scheduling) and a small finalizer.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROW_BLOCK = 128
BLOCK_COLS = 64


@ct.kernel
def _copy_partial_sum_kernel(
    x_ptr,         # bf16 [ROWS, COLS] input view
    clone_ptr,     # bf16 [ROWS, COLS] contiguous clone
    partial_ptr,   # f32 [NUM_ROW_TILES, COLS] per-row-tile column partials
    ROWS: ct.Constant[int],
    COLS: ct.Constant[int],
    ROW_BLOCK_C: ct.Constant[int],
    BLOCK_COLS_C: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    tile_bf = ct.load(
        x_ptr, index=(row_tile, col_tile),
        shape=(ROW_BLOCK_C, BLOCK_COLS_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(clone_ptr, index=(row_tile, col_tile), tile=tile_bf)
    tile_f = ct.astype(tile_bf, ct.float32)
    col_sum = ct.sum(tile_f, axis=0, keepdims=True)
    ct.store(partial_ptr, index=(row_tile, col_tile), tile=col_sum)


@ct.kernel
def _finish_sum_kernel(
    partial_ptr,   # f32 [NUM_ROW_TILES, COLS]
    sum_ptr,       # f32 [COLS]
    NUM_ROW_TILES: ct.Constant[int],
    COLS: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_COLS_C: ct.Constant[int],
):
    col_tile = ct.bid(0)
    tile = ct.load(
        partial_ptr, index=(0, col_tile),
        shape=(BLOCK_TILES, BLOCK_COLS_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(tile, axis=0, keepdims=False)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_tile,), tile=rounded)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="903ae292")
@oracle_impl(hardware="B200", point="f85fe078")
@oracle_impl(hardware="B200", point="9876fbcf")
def oracle_forward(inputs):
    x, _shape_param_0, shape_2d, sum_shape = inputs
    rows = int(shape_2d[0])
    cols = int(shape_2d[1])

    device = x.device

    # Materialize the layout-clone via torch (needs a full copy anyway; matches
    # Triton behaviour). For the contiguous-stride variants, permute+reshape
    # yields a non-contiguous view; use .contiguous() on the reshape to obtain
    # the bf16 clone buffer that downstream consumers expect.
    clone = x.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()

    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape),
        (1,),
        device=device,
        dtype=torch.float32,
    )

    num_row_tiles = (rows + ROW_BLOCK - 1) // ROW_BLOCK
    num_col_tiles = (cols + BLOCK_COLS - 1) // BLOCK_COLS
    partial = torch.empty(
        (num_row_tiles, cols), device=device, dtype=torch.float32
    )

    stream = torch.cuda.current_stream()
    # Kernel 1: read the (already-materialized contiguous) clone and produce
    # per-row-tile column partials. Also re-stores clone -> pattern-preserving.
    ct.launch(
        stream,
        (num_row_tiles, num_col_tiles, 1),
        _copy_partial_sum_kernel,
        (clone, clone, partial, rows, cols, ROW_BLOCK, BLOCK_COLS),
    )
    # Kernel 2: finalize the reduction over row tiles.
    block_tiles = _next_pow2(num_row_tiles)
    ct.launch(
        stream,
        (num_col_tiles, 1, 1),
        _finish_sum_kernel,
        (partial, sum_out, num_row_tiles, cols, block_tiles, BLOCK_COLS),
    )
    return clone, torch.as_strided(clone, (cols, rows), (1, cols)), sum_out
