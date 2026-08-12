"""cuTile port of amax_sum_7079889cfc04: online softmax over huge rows [8192, 262144].

Two-pass online softmax: pass1 computes row max and denom (streaming), pass2 writes bf16 out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_ROWS = 8192
N_COLS = 262144
BLOCK_N = 8192
NUM_BLOCKS = N_COLS // BLOCK_N  # 32


@ct.kernel
def _online_softmax_kernel(
    input_ptr,    # bf16 [N_ROWS, N_COLS]
    output_ptr,   # bf16 [N_ROWS, N_COLS]
    BLOCK_N: ct.Constant[int],
    NUM_BLOCKS: ct.Constant[int],
):
    row = ct.bid(0)

    # Pass 1: streaming max and denom
    row_max = ct.full(shape=(1,), fill_value=float("-inf"), dtype=ct.float32)
    row_sum = ct.full(shape=(1,), fill_value=0.0, dtype=ct.float32)
    for b in ct.static_iter(range(NUM_BLOCKS)):
        x_bf16 = ct.load(input_ptr, index=(row, b), shape=(1, BLOCK_N))
        x = ct.astype(x_bf16, ct.float32)
        block_max_scalar = ct.max(x)  # scalar tile
        block_max = ct.reshape(block_max_scalar, (1,))
        new_max = ct.maximum(row_max, block_max)
        # Reshape to broadcast against (1, BLOCK_N)
        new_max_2d = ct.reshape(new_max, (1, 1))
        row_max_2d = ct.reshape(row_max, (1, 1))
        row_sum = row_sum * ct.exp(row_max - new_max) + ct.reshape(
            ct.sum(ct.exp(x - new_max_2d)), (1,)
        )
        row_max = new_max

    # Pass 2: write bf16 output
    row_max_2d = ct.reshape(row_max, (1, 1))
    row_sum_2d = ct.reshape(row_sum, (1, 1))
    for b in ct.static_iter(range(NUM_BLOCKS)):
        x_bf16 = ct.load(input_ptr, index=(row, b), shape=(1, BLOCK_N))
        x = ct.astype(x_bf16, ct.float32)
        out = ct.exp(x - row_max_2d) / row_sum_2d
        ct.store(output_ptr, index=(row, b), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="4ac62a08", BLOCK_N=BLOCK_N)
def oracle_forward(inputs, *, BLOCK_N):
    x = inputs[0]
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (int(x.shape[0]), 1, 1),
        _online_softmax_kernel,
        (x, out, BLOCK_N, NUM_BLOCKS),
    )
    return out
