"""cuTile port of amax_sum_c30c6c7569c8: TrOCR causal-mask softmax.

Produces:
  where: bf16[64, 1, 256, 256] — causal mask (0 for k<=q, -3.39e38 for k>q).
  probs: bf16[1024, 256, 256] — softmax(scores + mask.broadcast, dim=-1).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BF16_MIN = -3.3895313892515355e38
BATCH = 64
HEADS = 16
Q_LEN = 256
K_LEN = 256
BLOCK_Q = 8
SOFTMAX_BLOCK_Q = 8


@ct.kernel
def _causal_mask_kernel(
    mask_out,  # bf16 [BATCH, Q_LEN, K_LEN]
    Q_LEN_c: ct.Constant[int],
    K_LEN_c: ct.Constant[int],
    BLOCK_Q_c: ct.Constant[int],
):
    b = ct.bid(0)
    qb = ct.bid(1)
    q = ct.arange(BLOCK_Q_c, dtype=ct.int32) + qb * BLOCK_Q_c
    k = ct.arange(K_LEN_c, dtype=ct.int32)
    q_2d = ct.reshape(q, (BLOCK_Q_c, 1))
    k_2d = ct.reshape(k, (1, K_LEN_c))
    causal = ct.less_equal(k_2d, q_2d)
    zeros = ct.zeros((BLOCK_Q_c, K_LEN_c), dtype=ct.bfloat16)
    fill = ct.full((BLOCK_Q_c, K_LEN_c), BF16_MIN, dtype=ct.bfloat16)
    values = ct.where(causal, zeros, fill)
    values_3d = ct.reshape(values, (1, BLOCK_Q_c, K_LEN_c))
    ct.store(mask_out, index=(b, qb, 0), tile=values_3d)


@ct.kernel
def _softmax_row_kernel(
    x,      # bf16 [BATCH*HEADS, Q_LEN, K_LEN]
    mask,   # bf16 [BATCH, Q_LEN, K_LEN]
    out,    # bf16 [BATCH*HEADS, Q_LEN, K_LEN]
    HEADS_c: ct.Constant[int],
    Q_LEN_c: ct.Constant[int],
    K_LEN_c: ct.Constant[int],
    BLOCK_Q_c: ct.Constant[int],
):
    bh = ct.bid(0)
    qb = ct.bid(1)
    b = bh // HEADS_c
    # Load a (BLOCK_Q, K_LEN) tile at once — many rows per program to
    # amortize launch overhead. Rows are independent softmax rows.
    # Note: index is in block units (multiplied by tile shape internally).
    x_tile = ct.load(x, index=(bh, qb, 0), shape=(1, BLOCK_Q_c, K_LEN_c))
    mask_tile = ct.load(mask, index=(b, qb, 0), shape=(1, BLOCK_Q_c, K_LEN_c))
    x_2d = ct.reshape(x_tile, (BLOCK_Q_c, K_LEN_c))
    mask_2d = ct.reshape(mask_tile, (BLOCK_Q_c, K_LEN_c))
    x_f = ct.astype(x_2d, ct.float32)
    mask_f = ct.astype(mask_2d, ct.float32)
    # Score is (x + mask) with bf16 rounding
    score_bf16 = ct.astype(x_f + mask_f, ct.bfloat16)
    score_f = ct.astype(score_bf16, ct.float32)
    row_max = ct.max(score_f, axis=1, keepdims=True)
    exp_v = ct.exp(score_f - row_max)
    denom = ct.sum(exp_v, axis=1, keepdims=True)
    probs = exp_v / denom
    probs_3d = ct.reshape(ct.astype(probs, ct.bfloat16), (1, BLOCK_Q_c, K_LEN_c))
    ct.store(out, index=(bh, qb, 0), tile=probs_3d)


@oracle_impl(hardware="B200", point="33572d5f")
def oracle_forward(inputs):
    arg0_1, _shape0, _shape1, _shape2 = inputs
    # arg0 shape: bf16[1024, 256, 256]
    mask = torch.empty(
        (BATCH, 1, Q_LEN, K_LEN),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty(
        (BATCH * HEADS, Q_LEN, K_LEN),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()

    # Kernel 1: build mask
    mask_3d = mask.view(BATCH, Q_LEN, K_LEN)
    ct.launch(
        stream,
        (BATCH, Q_LEN // BLOCK_Q, 1),
        _causal_mask_kernel,
        (mask_3d, Q_LEN, K_LEN, BLOCK_Q),
    )
    # Kernel 2: softmax over rows, batched SOFTMAX_BLOCK_Q rows per program
    # to reduce launch count from (BATCH*HEADS*Q_LEN) to
    # (BATCH*HEADS*Q_LEN/SOFTMAX_BLOCK_Q).
    x_3d = arg0_1.view(BATCH * HEADS, Q_LEN, K_LEN)
    ct.launch(
        stream,
        (BATCH * HEADS, Q_LEN // SOFTMAX_BLOCK_Q, 1),
        _softmax_row_kernel,
        (x_3d, mask_3d, out, HEADS, Q_LEN, K_LEN, SOFTMAX_BLOCK_Q),
    )
    return mask, out
