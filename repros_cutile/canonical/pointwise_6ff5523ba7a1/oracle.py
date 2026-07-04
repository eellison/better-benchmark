"""cuTile port of pointwise_6ff5523ba7a1: causal mask materialization.

Combines a per-batch source mask (f32 -> bool) with the k <= q causal
predicate to produce bool[batch, 1, seq, seq].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _causal_mask_kernel(
    mask,    # f32 [batch, seq]
    out,     # bool [batch, 1, seq, seq]
    SEQ: ct.Constant[int],
    BLOCK_Q: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    batch = ct.bid(0)
    q_tile = ct.bid(1)
    k_tile = ct.bid(2)

    # Load source mask row for this batch, k-tile: shape (1, BLOCK_K)
    src = ct.load(mask, index=(batch, k_tile), shape=(1, BLOCK_K))
    src_bool = src != 0.0  # (1, BLOCK_K)

    # Build q and k coordinate tiles
    q_ids = ct.arange(BLOCK_Q, dtype=ct.int32) + q_tile * BLOCK_Q  # (BLOCK_Q,)
    k_ids = ct.arange(BLOCK_K, dtype=ct.int32) + k_tile * BLOCK_K  # (BLOCK_K,)
    q_2d = ct.reshape(q_ids, (BLOCK_Q, 1))
    k_2d = ct.reshape(k_ids, (1, BLOCK_K))
    causal = k_2d <= q_2d  # (BLOCK_Q, BLOCK_K)

    # Broadcast src_bool (1, BLOCK_K) to (BLOCK_Q, BLOCK_K)
    src_broadcast = ct.reshape(src_bool, (1, BLOCK_K))
    out_val = causal & src_broadcast  # (BLOCK_Q, BLOCK_K)
    out_val = ct.reshape(out_val, (1, 1, BLOCK_Q, BLOCK_K))
    ct.store(out, index=(batch, 0, q_tile, k_tile), tile=out_val)


@oracle_impl(hardware="B200", point="003d4361", BLOCK_Q=32, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_Q, BLOCK_K):
    source_mask, expand_shape = inputs
    batch = int(source_mask.shape[0])
    seq = int(source_mask.shape[1])
    base = torch.empty_strided(
        (batch, 1, seq, seq),
        (seq * seq, seq * seq, seq, 1),
        device=source_mask.device,
        dtype=torch.bool,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, seq // BLOCK_Q, seq // BLOCK_K),
        _causal_mask_kernel,
        (source_mask, base, seq, BLOCK_Q, BLOCK_K),
    )
    return base.expand(tuple(int(dim) for dim in expand_shape))
