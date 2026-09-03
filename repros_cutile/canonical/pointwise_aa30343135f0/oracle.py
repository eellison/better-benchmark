"""cuTile port of pointwise_aa30343135f0: segmented causal attention mask (6 outputs).

For each (batch, q, k): output = 0.0 if (k <= q) and segment[b,q] == segment[b,k], else -inf.
Compute once, then clone 5 times.
"""

import math

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512


@ct.kernel
def _segmented_causal_mask_kernel(
    segments,     # (BATCH, SEQ) int64
    out,          # (BATCH, 1, SEQ, SEQ) bf16
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    NEG_INF: ct.Constant[float],
):
    batch = ct.bid(0)
    q_block = ct.bid(1)
    k_block = ct.bid(2)

    q_segs = ct.load(segments, index=(batch, q_block), shape=(1, BLOCK_M))
    k_segs = ct.load(segments, index=(batch, k_block), shape=(1, BLOCK_N))

    q_ids = ct.arange(BLOCK_M, dtype=ct.int32) + q_block * BLOCK_M
    k_ids = ct.arange(BLOCK_N, dtype=ct.int32) + k_block * BLOCK_N

    q_2d = ct.reshape(q_ids, (BLOCK_M, 1))
    k_2d = ct.reshape(k_ids, (1, BLOCK_N))
    q_segs_2d = ct.reshape(q_segs, (BLOCK_M, 1))
    k_segs_2d = ct.reshape(k_segs, (1, BLOCK_N))

    keep = (k_2d <= q_2d) & (q_segs_2d == k_segs_2d)
    zero_tile = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.bfloat16)
    ninf_tile = ct.full((BLOCK_M, BLOCK_N), fill_value=NEG_INF, dtype=ct.bfloat16)
    values = ct.where(keep, zero_tile, ninf_tile)
    values_4d = ct.reshape(values, (1, 1, BLOCK_M, BLOCK_N))
    ct.store(out, index=(batch, 0, q_block, k_block), tile=values_4d)


@oracle_impl(hardware="B200", point="26cc4258")
def oracle_forward(inputs):
    segments, _shape_param_0 = inputs
    out0 = torch.empty_strided(
        (BATCH, 1, SEQ, SEQ),
        (SEQ * SEQ, SEQ * SEQ, SEQ, 1),
        device=segments.device,
        dtype=torch.bfloat16,
    )
    BLOCK_M = 16
    BLOCK_N = 64
    neg_inf = -math.inf
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, SEQ // BLOCK_M, SEQ // BLOCK_N),
        _segmented_causal_mask_kernel,
        (segments, out0, BLOCK_M, BLOCK_N, neg_inf),
    )
    # Six identical outputs.
    return (out0, out0.clone(), out0.clone(), out0.clone(), out0.clone(), out0.clone())
