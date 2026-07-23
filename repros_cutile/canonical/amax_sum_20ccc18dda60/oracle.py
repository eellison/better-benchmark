"""cuTile port of amax_sum_20ccc18dda60 (NEW_PATTERN): bf16 broadcast-add
attention softmax [1024,256,256] with a [64,1,256,256] bias broadcast per-head.

Match Triton's BLOCK_M=8 tiling. BLOCK_M=8 divides Q_LEN=256, so each row
block lives fully within a single (batch, head) — thus the bias tile is a
normal contiguous cuTile load without advanced indexing.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HEADS = 16
Q_LEN = 256
K_LEN = 256


@ct.kernel
def _broadcast_add_softmax_kernel(
    x_2d,          # bf16 [N_ROWS, K_LEN] view of x (1024*256, 256)
    bias_4d,       # bf16 [64, 1, Q_LEN, K_LEN]
    out_2d,        # bf16 [N_ROWS, K_LEN]
    HEADS_C: ct.Constant[int],
    Q_LEN_C: ct.Constant[int],
    Q_BLOCKS_PER_HEAD: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    # BLOCK_M rows in one tile — all share the same (batch, head, q_block).
    flat_bh = row_block // Q_BLOCKS_PER_HEAD
    batch = flat_bh // HEADS_C
    q_block = row_block - flat_bh * Q_BLOCKS_PER_HEAD  # tile-space q index

    x_bf = ct.load(x_2d, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    x = ct.astype(x_bf, ct.float32)

    # Load bias tile [BLOCK_M, BLOCK_N] at (batch, 0, q_block*BLOCK_M, 0)
    # using tile-space indexing.
    bias_4d_tile = ct.load(bias_4d, index=(batch, 0, q_block, 0),
                           shape=(1, 1, BLOCK_M, BLOCK_N))
    bias = ct.reshape(bias_4d_tile, (BLOCK_M, BLOCK_N))
    bias_f = ct.astype(bias, ct.float32)

    scores = x + bias_f
    row_max = ct.max(scores, axis=1, keepdims=True)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(out_2d, index=(row_block, 0),
             tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="38831ee4", BLOCK_M=8, BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    # x is [1024,256,256], out same shape/stride.
    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_rows = int(arg0_1.shape[0] * arg0_1.shape[1])
    x_2d = arg0_1.view(n_rows, K_LEN)
    out_2d = out.view(n_rows, K_LEN)
    q_blocks_per_head = Q_LEN // BLOCK_M

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _broadcast_add_softmax_kernel,
        (x_2d, arg1_1, out_2d, HEADS, Q_LEN, q_blocks_per_head, BLOCK_M, BLOCK_N),
    )
    return out
