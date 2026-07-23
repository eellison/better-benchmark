"""cuTile port of sum_8ef711188ca0: singleton select_scatter materialize + column sum.

Input arg0_1: bf16[128, 768]. Full+select_scatter equals arg0_1 stored into
`[128, 1, 768]`. Also compute column sum with bf16 rounding boundary.
Returns (select_scatter, view, permute, sum_out).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
COLS = 768


@ct.kernel
def _materialize_sum_kernel(
    x_ptr,       # bf16 [ROWS, COLS]
    base_ptr,    # bf16 [ROWS, COLS]
    sum_ptr,     # f32 [COLS]
    BLOCK_N: ct.Constant[int],
    ROWS_: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(x_ptr, index=(0, col_block), shape=(ROWS_, BLOCK_N))
    ct.store(base_ptr, index=(0, col_block), tile=values)

    values_f = ct.astype(values, ct.float32)
    totals = ct.sum(values_f, axis=0)
    rounded = ct.astype(ct.astype(totals, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=rounded)


@oracle_impl(hardware="B200", point="9c23c094", BLOCK_N=8)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    select_scatter = torch.empty_strided(
        (ROWS, 1, COLS),
        (COLS, COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((COLS,), (1,), device=x.device, dtype=torch.float32)

    select_2d = select_scatter.view(ROWS, COLS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(COLS, BLOCK_N), 1, 1),
        _materialize_sum_kernel,
        (x, select_2d, sum_out, BLOCK_N, ROWS),
    )

    view = select_scatter.view(ROWS, COLS)
    return select_scatter, view, view.permute(1, 0), sum_out
