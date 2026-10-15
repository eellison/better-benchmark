"""cuTile port of sum_b7a98f3c6228: DeepRecommender bf16 SELU-style producer
with right-pad + transpose-pad materializations plus a bf16-round-tripped
column sum.

The Triton oracle uses three kernels (materialize, transpose+sum, zero-pad-
tail). Here we do it all in one cuTile kernel per column block: for each
tile of BLOCK_N contiguous cols, load 256 rows of x and gate with ZERO
padding for OOB cols, compute the SELU-like value in fp32, cast to bf16,
store to both the right-pad and transposed layouts, then column-sum with a
bf16 round-trip.

Because the pad column (col=COLS) reads x=0 and gate=0 with ZERO padding,
its computed value_bf16 is exactly 0.0, matching the Triton oracle's
explicit _zero_pad_tail_kernel writes into the pad col/row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 256
COLS = 197951
PADDED_COLS = 197952


@ct.kernel
def _selu_materialize_kernel(
    x_ptr,           # bf16 [ROWS, COLS]
    gate_ptr,        # bf16 [ROWS, COLS]   stride (PADDED_COLS, 1)
    right_pad_ptr,   # bf16 [ROWS, PADDED_COLS]
    trans_pad_ptr,   # bf16 [PADDED_COLS, ROWS]
    sum_out_ptr,     # f32  [COLS]
    ROWS_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    c_blk = ct.bid(0)

    x_bf = ct.load(
        x_ptr, index=(0, c_blk), shape=(ROWS_C, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    gate_bf = ct.load(
        gate_ptr, index=(0, c_blk), shape=(ROWS_C, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)
    gate = ct.astype(gate_bf, ct.float32)

    neg = x * 1.7580993408473766 * ct.exp(gate)
    pos = x * 1.0507009873554805
    value = ct.where(gate <= 0.0, neg, pos)
    value_bf16 = ct.astype(value, ct.bfloat16)

    # Store right-pad layout: right_pad[r, c_blk*BLOCK_N + i] = value[r, i].
    ct.store(right_pad_ptr, index=(0, c_blk), tile=value_bf16)

    # Store transpose layout: trans_pad[c_blk*BLOCK_N + i, r] = value[r, i].
    value_t = ct.transpose(value_bf16)  # (BLOCK_N, ROWS_C)
    ct.store(trans_pad_ptr, index=(c_blk, 0), tile=value_t)

    # Column sum with bf16 round-trip to match the Triton oracle exactly.
    total = ct.sum(ct.astype(value_bf16, ct.float32), axis=0)  # (BLOCK_N,)
    total_bf16 = ct.astype(total, ct.bfloat16)
    total_f32 = ct.astype(total_bf16, ct.float32)
    # sum_out is length COLS; OOB tiles (last col-block) are masked away by cuTile.
    ct.store(sum_out_ptr, index=(c_blk,), tile=total_f32)


@oracle_impl(
    hardware="B200",
    point="348dd978",
    BLOCK_N=64,
)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, gate, _shape = inputs
    del _shape

    device = x.device
    right_pad = torch.empty_strided(
        (ROWS, PADDED_COLS),
        (PADDED_COLS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    trans_pad = torch.empty_strided(
        (PADDED_COLS, ROWS),
        (ROWS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        (COLS,),
        (1,),
        device=device,
        dtype=torch.float32,
    )

    num_col_blocks = (PADDED_COLS + BLOCK_N - 1) // BLOCK_N
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_col_blocks, 1, 1),
        _selu_materialize_kernel,
        (x, gate, right_pad, trans_pad, sum_out, ROWS, BLOCK_N),
    )

    return right_pad, trans_pad, sum_out
