"""cuTile port of pointwise_6d75f431992b: segment causal attention mask.

For each (batch, q, k):
  q_seg = cumsum[batch, positions[q]]
  k_seg = cumsum[batch, positions[k]]
  base[batch, 0, q, k] = 0 if (k <= q AND q_seg == k_seg) else -inf
Returned as (base, base.expand(BATCH, HEADS, SEQ, SEQ)).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 12
SEQ = 512
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)

Q_TILE = 32
K_TILE = 32


@ct.kernel
def _segment_causal_kernel(
    positions,     # [SEQ] i64
    cumsum,        # [BATCH, SEQ] i64
    out,           # [BATCH, SEQ, SEQ] bf16 (viewed from [B,1,S,S])
    Q_TILE_c: ct.Constant[int],
    K_TILE_c: ct.Constant[int],
):
    b = ct.bid(0)
    qb = ct.bid(1)
    kb = ct.bid(2)

    q_pos = ct.load(positions, index=(qb,), shape=(Q_TILE_c,))
    k_pos = ct.load(positions, index=(kb,), shape=(K_TILE_c,))

    b_tile = ct.full((), b, dtype=ct.int64)
    q_seg = ct.gather(cumsum, (b_tile, q_pos))  # [Q_TILE]
    k_seg = ct.gather(cumsum, (b_tile, k_pos))  # [K_TILE]

    q_pos_2d = ct.reshape(q_pos, (Q_TILE_c, 1))
    k_pos_2d = ct.reshape(k_pos, (1, K_TILE_c))
    q_seg_2d = ct.reshape(q_seg, (Q_TILE_c, 1))
    k_seg_2d = ct.reshape(k_seg, (1, K_TILE_c))

    causal = ct.less_equal(k_pos_2d, q_pos_2d)
    same_seg = ct.equal(q_seg_2d, k_seg_2d)
    keep = ct.bitwise_and(causal, same_seg)

    zeros = ct.zeros((Q_TILE_c, K_TILE_c), dtype=ct.bfloat16)
    neg_inf = ct.full((Q_TILE_c, K_TILE_c), -float("inf"), dtype=ct.bfloat16)
    values = ct.where(keep, zeros, neg_inf)
    values_3d = ct.reshape(values, (1, Q_TILE_c, K_TILE_c))
    ct.store(out, index=(b, qb, kb), tile=values_3d)


@oracle_impl(hardware="B200", point="9210619b")
def oracle_forward(inputs):
    positions, cumsum, _shape0, _shape1 = inputs
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=positions.device,
        dtype=torch.bfloat16,
    )
    # positions is i64[1, 512] — 1D view
    positions_1d = positions.view(SEQ)
    # base has shape [B, 1, S, S] — flatten to [B, S, S] for kernel
    base_3d = base.view(BATCH, SEQ, SEQ)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, SEQ // Q_TILE, SEQ // K_TILE),
        _segment_causal_kernel,
        (positions_1d, cumsum, base_3d, Q_TILE, K_TILE),
    )
    return base, base.expand(EXPAND_SHAPE)
