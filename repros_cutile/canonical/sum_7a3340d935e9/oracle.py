"""cuTile port of sum_7a3340d935e9: (x*y).sum(dim=[0,1]) column reduction.

Iterates one column tile per program, accumulates x*y over the R = B*S row
dimension. Tile shape is chosen to divide both cols (768/1024) evenly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _column_prod_sum_kernel(
    x_ptr,           # f32 [R, C]
    y_ptr,           # f32 [R, C]
    out_ptr,         # f32 [C]
    R: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
):
    col_pid = ct.bid(0)
    n_row_tiles: ct.Constant[int] = R // ROW_BLOCK
    acc = ct.zeros((ROW_BLOCK, COL_BLOCK), dtype=ct.float32)
    for r in range(n_row_tiles):
        x = ct.load(x_ptr, index=(r, col_pid), shape=(ROW_BLOCK, COL_BLOCK))
        y = ct.load(y_ptr, index=(r, col_pid), shape=(ROW_BLOCK, COL_BLOCK))
        acc = acc + x * y
    total = ct.sum(acc, axis=0)
    ct.store(out_ptr, index=(col_pid,), tile=total)


@oracle_impl(hardware="B200", point="c57eca43")
@oracle_impl(hardware="B200", point="b79fa38b")
@oracle_impl(hardware="B200", point="fc97d3ab")
@oracle_impl(hardware="B200", point="f4eac48c")
def oracle_forward(inputs):
    x, y = inputs
    rows = int(x.shape[0]) * int(x.shape[1])
    cols = int(x.shape[2])
    x2 = x.reshape(rows, cols).contiguous()
    y2 = y.reshape(rows, cols).contiguous()
    out = torch.empty_strided((cols,), (1,), device=x.device, dtype=torch.float32)

    # Choose block sizes: 32 divides all our col options (768, 1024)
    COL_BLOCK = 32
    # Row block: choose a pow2 that divides rows
    if rows % 128 == 0:
        ROW_BLOCK = 128
    elif rows % 64 == 0:
        ROW_BLOCK = 64
    elif rows % 32 == 0:
        ROW_BLOCK = 32
    elif rows % 16 == 0:
        ROW_BLOCK = 16
    else:
        ROW_BLOCK = 8

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (cols // COL_BLOCK, 1, 1),
        _column_prod_sum_kernel,
        (x2, y2, out, rows, ROW_BLOCK, COL_BLOCK),
    )
    return out
