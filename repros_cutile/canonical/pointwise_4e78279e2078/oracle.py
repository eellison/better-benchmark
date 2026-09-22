"""cuTile port of pointwise_4e78279e2078: BART causal + source-mask materialization.

For each (batch, q, k) position, the output is (k <= q) & (source_mask[batch, k] != 0).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bart_causal_mask_kernel(
    mask_ptr,       # (batch, seq) f32
    out_ptr,        # (batch, 1, seq, seq) bool
    BLOCK_Q: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    batch = ct.bid(0)
    q_block = ct.bid(1)
    k_block = ct.bid(2)

    # Load source mask row (batch, 0), a (1, BLOCK_K) tile.
    src = ct.load(mask_ptr, index=(batch, k_block), shape=(1, BLOCK_K))
    src_bool = src != 0.0  # (1, BLOCK_K)

    q_ids = ct.arange(BLOCK_Q, dtype=ct.int32) + q_block * BLOCK_Q
    k_ids = ct.arange(BLOCK_K, dtype=ct.int32) + k_block * BLOCK_K
    q_2d = ct.reshape(q_ids, (BLOCK_Q, 1))
    k_2d = ct.reshape(k_ids, (1, BLOCK_K))

    causal = k_2d <= q_2d  # (BLOCK_Q, BLOCK_K)
    result = causal & ct.reshape(src_bool, (1, BLOCK_K))

    # Store into (batch, 0, q_block, k_block) of the (batch, 1, seq, seq) tile.
    out_4d = ct.reshape(result, (1, 1, BLOCK_Q, BLOCK_K))
    ct.store(out_ptr, index=(batch, 0, q_block, k_block), tile=out_4d)


@oracle_impl(hardware="B200", point="5002b631", BLOCK_Q=32, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_Q: int, BLOCK_K: int):
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
        (batch, ct.cdiv(seq, BLOCK_Q), ct.cdiv(seq, BLOCK_K)),
        _bart_causal_mask_kernel,
        (source_mask, base, BLOCK_Q, BLOCK_K),
    )
    return base.expand(tuple(int(dim) for dim in expand_shape))
