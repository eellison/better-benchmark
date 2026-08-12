"""cuTile port of amax_sum_sum_4325c394b077: large-row bf16 softmax + sum.

For each row of the [8192, 262144] input, we do two passes:
1. Online-softmax to compute the row max and the row exp-sum denominator.
2. Recompute the bf16-rounded softmax value and accumulate a per-row scalar
   sum.
Then a small final kernel reduces the 8192 per-row sums into a single bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 262144


@ct.kernel
def _softmax_stats_kernel(
    x_ptr,           # bf16 (ROWS, COLS)
    amax_ptr,        # fp32 (ROWS,)
    denom_ptr,       # fp32 (ROWS,)
    row_sums_ptr,    # fp32 (ROWS,)
    BLOCK_N: ct.Constant[int],
    N_TILES: ct.Constant[int],
):
    row = ct.bid(0)

    row_max = ct.astype(-1.0e38, ct.float32)
    denom = ct.astype(0.0, ct.float32)

    for block_idx in range(N_TILES):
        x = ct.load(x_ptr, index=(row, block_idx), shape=(1, BLOCK_N))
        x_f = ct.astype(x, ct.float32)
        x_f = ct.reshape(x_f, (BLOCK_N,))
        block_max = ct.max(x_f)
        next_max = ct.where(block_max > row_max, block_max, row_max)
        denom = denom * ct.exp(row_max - next_max) + ct.sum(ct.exp(x_f - next_max))
        row_max = next_max

    row_sum = ct.astype(0.0, ct.float32)
    for block_idx in range(N_TILES):
        x = ct.load(x_ptr, index=(row, block_idx), shape=(1, BLOCK_N))
        x_f = ct.astype(x, ct.float32)
        x_f = ct.reshape(x_f, (BLOCK_N,))
        probs = ct.exp(x_f - row_max) / denom
        probs_bf = ct.astype(ct.astype(probs, ct.bfloat16), ct.float32)
        row_sum = row_sum + ct.sum(probs_bf)

    # Store per-row scalars (shape (1,) so cuTile is happy).
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    ct.store(row_sums_ptr, index=(row,), tile=ct.reshape(row_sum, (1,)))


@ct.kernel
def _final_sum_kernel(
    row_sums_ptr,    # fp32 (ROWS,)
    out_ptr,         # bf16 ()
    BLOCK_M: ct.Constant[int],
):
    vals = ct.load(row_sums_ptr, index=(0,), shape=(BLOCK_M,),
                   padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(vals)
    total_bf = ct.astype(total, ct.bfloat16)
    ct.store(out_ptr, index=(0,), tile=ct.reshape(total_bf, (1,)))


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@oracle_impl(hardware="B200", point="4ac62a08", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    (x,) = inputs
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    assert rows == ROWS and cols == COLS

    amax = torch.empty_strided(
        (rows, 1), (1, 1), device=x.device, dtype=torch.float32
    )
    denom = torch.empty_strided(
        (rows, 1), (1, 1), device=x.device, dtype=torch.float32
    )
    row_sums = torch.empty((rows,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)

    n_tiles = cols // BLOCK_N
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _softmax_stats_kernel,
        (
            x,
            amax.view(rows),
            denom.view(rows),
            row_sums,
            BLOCK_N,
            n_tiles,
        ),
    )

    block_m = _next_pow2(rows)
    ct.launch(
        stream,
        (1, 1, 1),
        _final_sum_kernel,
        (row_sums, out.view(1), block_m),
    )
    return amax, denom, out
