"""cuTile port of sum_76127d6ec415: DeiT bf16 divide + column sum + roundtrip.

Compute div = arg0 / 2 (bf16), materialize [128, 1000] bf16 out; then take
column sum in fp32 with bf16 rounding boundary, duplicated into two f32[1000]
outputs. Return (div, div.permute(1,0), sum0, sum1).

COLS=1000 is not power of 2, but 1000 = 8*125 so BLOCK_N=8 tiles the columns
evenly. ROWS=128 is a power of 2, so the row axis fits in one tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
COLS = 1000
BLOCK_N_CT = 8  # divides 1000 evenly (125 tiles)


@ct.kernel
def _divide_sum_kernel(
    x_ptr,      # bf16 [ROWS, COLS]
    div_ptr,    # bf16 [ROWS, COLS]
    sum0_ptr,   # f32 [COLS]
    sum1_ptr,   # f32 [COLS]
    M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    x = ct.load(x_ptr, index=(0, col_block), shape=(M, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    div_f = x_f * 0.5
    div_bf16 = ct.astype(div_f, ct.bfloat16)
    ct.store(div_ptr, index=(0, col_block), tile=div_bf16)

    div_f_recov = ct.astype(div_bf16, ct.float32)
    col_sum = ct.sum(div_f_recov, axis=0)
    rounded = ct.astype(ct.astype(col_sum, ct.bfloat16), ct.float32)
    ct.store(sum0_ptr, index=(col_block,), tile=rounded)
    ct.store(sum1_ptr, index=(col_block,), tile=rounded)


@oracle_impl(hardware="B200", point="e781aba9")
def oracle_forward(inputs):
    arg0_1, _shape_param_0 = inputs
    del _shape_param_0

    div = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum0 = torch.empty_strided((COLS,), (1,), device=arg0_1.device, dtype=torch.float32)
    sum1 = torch.empty_strided((COLS,), (1,), device=arg0_1.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (COLS // BLOCK_N_CT, 1, 1),
        _divide_sum_kernel,
        (arg0_1, div, sum0, sum1, ROWS, BLOCK_N_CT),
    )
    return div, div.permute(1, 0), sum0, sum1
