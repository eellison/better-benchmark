"""cuTile port of pointwise_cbb8d3f0ed92: Blenderbot row index-put accumulate.

Zeros out a dense f32[128,2560] and atomically adds each source row into the
destination row given by index[src_row].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
COLS = 2560


@ct.kernel
def _row_scatter_kernel(
    index_arr,    # (128,) i64
    values,       # (128, 2560) f32
    out,          # (128, 2560) f32 (zero-initialized)
    ROWS_: ct.Constant[int],
    COLS_: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    src_row = ct.bid(0)
    col_block = ct.bid(1)
    # Load one index
    raw_index = ct.load(index_arr, index=(src_row,), shape=(1,))
    # Wrap negative index
    wrapped = ct.where(raw_index < 0, raw_index + ROWS_, raw_index)
    # Load values row
    vals = ct.load(values, index=(src_row, col_block), shape=(1, BLOCK_COLS))
    # Build broadcast index for scatter
    cols = ct.arange(BLOCK_COLS, dtype=ct.int64) + col_block * BLOCK_COLS
    row_idx = ct.reshape(wrapped, (1, 1))
    col_idx = ct.reshape(cols, (1, BLOCK_COLS))
    # atomic_add to out[wrapped, cols] += vals
    ct.atomic_add(out, (row_idx, col_idx), vals)


@oracle_impl(hardware="B200", point="154a9c83", BLOCK_COLS=256)
def oracle_forward(inputs, *, BLOCK_COLS):
    index, values, _shape0, _shape1 = inputs
    out = torch.zeros((ROWS, COLS), device=values.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ROWS, ct.cdiv(COLS, BLOCK_COLS), 1),
              _row_scatter_kernel,
              (index, values, out, ROWS, COLS, BLOCK_COLS))
    return out
