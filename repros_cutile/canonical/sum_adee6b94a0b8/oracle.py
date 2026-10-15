"""cuTile port of sum_adee6b94a0b8: DeepRecommender SELU + column sum.

Reads bf16 x and gate `[256, 512]`, computes SELU-like value, stores bf16
transposed clone, and returns f32 column sum. Skip the base transpose alias
which is returned as an aliased view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 256
COLS = 512


@ct.kernel
def _selu_sum_kernel(
    x_ptr,      # bf16 (ROWS, COLS)
    gate_ptr,   # bf16 (ROWS, COLS)
    base_ptr,   # bf16 (ROWS, COLS)
    sum_ptr,    # f32 (COLS,)
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Load a (ROWS, BLOCK_N) tile
    x = ct.load(x_ptr, index=(0, col_block), shape=(ROWS, BLOCK_N))
    gate = ct.load(gate_ptr, index=(0, col_block), shape=(ROWS, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    gate_f = ct.astype(gate, ct.float32)

    negative = (x_f * 1.0) * 1.7580993408473766
    negative = negative * ct.exp(gate_f * 1.0)
    positive = x_f * 1.0507009873554805
    value = ct.where(gate_f <= 0.0, negative, positive)
    value_bf16 = ct.astype(value, ct.bfloat16)
    ct.store(base_ptr, index=(0, col_block), tile=value_bf16)

    col_sum = ct.sum(ct.astype(value_bf16, ct.float32), axis=0)  # (BLOCK_N,)
    rounded_sum = ct.astype(ct.astype(col_sum, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=rounded_sum)


@oracle_impl(hardware="B200", point="a7f39cdb", BLOCK_N=2)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, gate, shape_param = inputs

    base = torch.empty_strided((ROWS, COLS), (COLS, 1), device=x.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((COLS + BLOCK_N - 1) // BLOCK_N, 1, 1),
        _selu_sum_kernel,
        (x, gate, base, sum_out, BLOCK_N),
    )
    return base.permute(1, 0), sum_out
