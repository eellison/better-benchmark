"""cuTile port of sum_977aa8760e55: bf16 [128,768] dim-0 sum to f32 [1,1,768].

NEW_PATTERN: sum bf16 [128, 768] over dim 0 with fp32 accumulation, output
f32 [1, 1, 768] contiguous.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
COLS = 768


@ct.kernel
def _sum_rows_kernel(
    x_ptr,      # bf16 [ROWS, COLS]
    out_ptr,    # f32 [COLS]
    ROWS_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Load (ROWS, BLOCK_N) tile as bf16
    values = ct.load(x_ptr, index=(0, col_block), shape=(ROWS_, BLOCK_N))
    values_f = ct.astype(values, ct.float32)
    totals = ct.sum(values_f, axis=0)   # (BLOCK_N,)
    ct.store(out_ptr, index=(col_block,), tile=totals)


@oracle_impl(hardware="B200", point="9c23c094", BLOCK_N=8)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, _shape_param = inputs
    out = torch.empty_strided(
        (1, 1, COLS),
        (COLS, COLS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out_1d = out.view(COLS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (COLS // BLOCK_N, 1, 1),
        _sum_rows_kernel,
        (x, out_1d, ROWS, BLOCK_N),
    )
    return out
