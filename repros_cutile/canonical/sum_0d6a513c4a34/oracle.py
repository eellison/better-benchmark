"""cuTile port of sum_0d6a513c4a34: hardswish activation + channel sum.

Compute the hardswish of x with gate = arg1_1 (bf16), store both the bf16
activation and the fp32 channel sum. Also return the f32 zero scalar.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 512
COLS = 1280
OUT_SHAPE = (ROWS, COLS, 1, 1)
OUT_STRIDE = (COLS, 1, 1, 1)


@ct.kernel
def _hardswish_materialize_sum_kernel(
    x_ptr,           # bf16 [ROWS, COLS]
    gate_ptr,        # bf16 [ROWS, COLS]
    activation_ptr,  # bf16 [ROWS, COLS]
    sum_ptr,         # fp32 [COLS]
    ROWS_C: ct.Constant[int],
    COLS_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)

    # Load the full column block at once: (ROWS, BLOCK_N)
    x = ct.load(x_ptr, index=(0, col_tile), shape=(ROWS_C, BLOCK_N))
    gate = ct.load(gate_ptr, index=(0, col_tile), shape=(ROWS_C, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    gate_f = ct.astype(gate, ct.float32)

    linear_gate = gate_f * (1.0 / 3.0) + 0.5
    middle = x_f * linear_gate
    thr_lt3 = ct.astype(ct.full((ROWS_C, BLOCK_N), 3.0, dtype=ct.float32), ct.float32)
    thr_le_neg3 = ct.astype(ct.full((ROWS_C, BLOCK_N), -3.0, dtype=ct.float32), ct.float32)
    value = ct.where(gate_f < thr_lt3, middle, x_f)
    zero_broadcast = ct.full((ROWS_C, BLOCK_N), 0.0, dtype=ct.float32)
    value = ct.where(gate_f <= thr_le_neg3, zero_broadcast, value)

    activation = ct.astype(value, ct.bfloat16)
    ct.store(activation_ptr, index=(0, col_tile), tile=activation)

    # Sum activation (bf16 -> f32) along axis 0 (rows) -> (BLOCK_N,)
    reduced = ct.sum(ct.astype(activation, ct.float32), axis=0)
    ct.store(sum_ptr, index=(col_tile,), tile=reduced)


@ct.kernel
def _zero_scalar_kernel(zero_ptr):
    ct.store(zero_ptr, index=(0,), tile=ct.zeros((1,), dtype=ct.float32))


@oracle_impl(hardware="B200", point="cf51a5e7", BLOCK_M=512, BLOCK_N=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    x, gate, _shape = inputs
    zero = torch.empty_strided((), (), device=x.device, dtype=torch.float32)
    activation = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.bfloat16,
    )
    sums = torch.empty_strided((COLS,), (1,), device=x.device, dtype=torch.float32)

    x_2d = x.view(ROWS, COLS)
    gate_2d = gate.view(ROWS, COLS)
    activation_2d = activation.view(ROWS, COLS)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _zero_scalar_kernel, (zero.view(1),))
    ct.launch(
        stream,
        (COLS // BLOCK_N, 1, 1),
        _hardswish_materialize_sum_kernel,
        (x_2d, gate_2d, activation_2d, sums, ROWS, COLS, BLOCK_N),
    )
    return zero, activation, sums
