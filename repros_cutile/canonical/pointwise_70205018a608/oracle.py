"""cuTile port of pointwise_70205018a608 (NEW_PATTERN):

Blenderbot causal mask: bool[batch, 1, S, S] where out[b, 0, q, k] =
(k <= q) AND (source_mask[b, k] != 0.0). Source mask is f32[batch, S].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _causal_mask_kernel(
    mask_ptr,  # f32 [batch, S]
    out_ptr,  # b8 [batch, 1, S, S]
    SEQ: ct.Constant[int],
    BLOCK_Q: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    batch = ct.bid(0)
    q_tile = ct.bid(1)
    k_tile = ct.bid(2)

    # Load source mask [batch, k_tile] tile of size BLOCK_K
    src = ct.load(mask_ptr, index=(batch, k_tile), shape=(1, BLOCK_K))
    src = ct.reshape(src, (BLOCK_K,))
    src_nonzero = src != 0.0

    # Build q and k coordinates.
    q_off = ct.arange(BLOCK_Q, dtype=ct.int32) + q_tile * BLOCK_Q
    k_off = ct.arange(BLOCK_K, dtype=ct.int32) + k_tile * BLOCK_K
    q_2d = ct.reshape(q_off, (BLOCK_Q, 1))
    k_2d = ct.reshape(k_off, (1, BLOCK_K))
    src_2d = ct.reshape(src_nonzero, (1, BLOCK_K))
    values = (k_2d <= q_2d) & src_2d
    # reshape to (1,1,BLOCK_Q,BLOCK_K) for the 4D store
    values = ct.reshape(values, (1, 1, BLOCK_Q, BLOCK_K))
    ct.store(out_ptr, index=(batch, 0, q_tile, k_tile), tile=values)


@oracle_impl(hardware="B200", point="228a3c6c", BLOCK_Q=32, BLOCK_K=128)
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
        (batch, (seq + BLOCK_Q - 1) // BLOCK_Q, (seq + BLOCK_K - 1) // BLOCK_K),
        _causal_mask_kernel,
        (source_mask, base, seq, BLOCK_Q, BLOCK_K),
    )
    return base.expand(tuple(int(dim) for dim in expand_shape))
