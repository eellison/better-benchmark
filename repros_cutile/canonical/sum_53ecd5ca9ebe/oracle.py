"""cuTile port of sum_53ecd5ca9ebe: SELU activation + column sum.

For each column block: compute SELU-style output (per repro constants) in fp32,
round to bf16, store bf16 output and the fp32 column sum with a bf16 rounding
boundary on the accumulated sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 256
COLS = 512


@ct.kernel
def _selu_bf16_sum_kernel(
    x_ptr,       # bf16 [ROWS, COLS]
    gate_ptr,    # bf16 [ROWS, COLS]
    out_ptr,     # bf16 [ROWS, COLS]
    sum_ptr,     # f32 [COLS]
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)
    x = ct.load(x_ptr, index=(0, col_tile), shape=(ROWS, BLOCK_N))
    gate = ct.load(gate_ptr, index=(0, col_tile), shape=(ROWS, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    gate_f = ct.astype(gate, ct.float32)

    neg = x_f * 1.7580993408473766 * ct.exp(gate_f)
    pos = x_f * 1.0507009873554805
    zero = ct.full((ROWS, BLOCK_N), 0.0, dtype=ct.float32)
    value = ct.where(gate_f <= zero, neg, pos)
    value_bf16 = ct.astype(value, ct.bfloat16)

    ct.store(out_ptr, index=(0, col_tile), tile=value_bf16)

    col_sum = ct.sum(ct.astype(value_bf16, ct.float32), axis=0)
    rounded = ct.astype(ct.astype(col_sum, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_tile,), tile=rounded)


@oracle_impl(hardware="B200", point="a7f39cdb", BLOCK_N=2)
def oracle_forward(inputs, *, BLOCK_N):
    x, gate, _shape = inputs
    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        (COLS,),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (COLS // BLOCK_N, 1, 1),
        _selu_bf16_sum_kernel,
        (x, gate, out, sum_out, BLOCK_N),
    )
    return out, out.permute(1, 0), sum_out
