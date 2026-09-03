"""cuTile port of sum_eb86a17deab9: fp32 column sum over [B*S, H] -> [H]."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _col_sum_kernel(
    x_ptr,       # [ROWS, COLS] contiguous
    out_ptr,     # [COLS]
    ROWS_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_tile = ct.bid(0)
    # Sum along rows, iterate row tiles inside the kernel
    acc = ct.zeros(shape=(BLOCK_C,), dtype=ct.float32)
    for row_tile in range(ct.cdiv(ROWS_C, BLOCK_R)):
        tile = ct.load(x_ptr, index=(row_tile, col_tile),
                       shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
        acc = acc + ct.sum(tile, axis=0)
    ct.store(out_ptr, index=(col_tile,), tile=acc)


@oracle_impl(hardware="B200", point="b85aeb78", BLOCK_ROWS=256, BLOCK_COLS=32)
@oracle_impl(hardware="B200", point="784a7239", BLOCK_ROWS=256, BLOCK_COLS=32)
@oracle_impl(hardware="B200", point="df1f991c", BLOCK_ROWS=256, BLOCK_COLS=32)
@oracle_impl(hardware="B200", point="0a855bca", BLOCK_ROWS=256, BLOCK_COLS=32)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_COLS: int):
    (x,) = inputs
    rows = int(x.shape[0]) * int(x.shape[1])
    cols = int(x.shape[2])
    out = torch.empty_strided((cols,), (1,), device=x.device, dtype=torch.float32)
    x2d = x.view(rows, cols)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(cols, BLOCK_COLS), 1, 1),
        _col_sum_kernel,
        (x2d, out, rows, BLOCK_ROWS, BLOCK_COLS),
    )
    return out
