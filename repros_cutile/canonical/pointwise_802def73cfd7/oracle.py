"""cuTile port of pointwise_802def73cfd7: causal-bias 12x multi-output fill.

For each element, if k <= q store 0.0 else -inf, into 12 output tensors.
"""

import math

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NEG_INF_BF16 = float("-inf")


@ct.kernel
def _causal_bias_pair_kernel(
    out_a,  # (4, 1, 2048, 2048) bf16
    out_b,
    BM: ct.Constant[int],
    BN: ct.Constant[int],
    SEQ: ct.Constant[int],
):
    bid_b = ct.bid(0)  # batch
    bid_i = ct.bid(1)  # query block index
    bid_j = ct.bid(2)  # key block index

    q_base = bid_i * BM
    k_base = bid_j * BN
    q_idx = q_base + ct.arange(BM, dtype=ct.int32)
    k_idx = k_base + ct.arange(BN, dtype=ct.int32)
    # broadcast to (BM, BN)
    q = ct.reshape(q_idx, (BM, 1))
    k = ct.reshape(k_idx, (1, BN))
    keep = k <= q
    zero_tile = ct.full((BM, BN), 0.0, dtype=ct.bfloat16)
    ninf_tile = ct.full((BM, BN), NEG_INF_BF16, dtype=ct.bfloat16)
    values = ct.where(keep, zero_tile, ninf_tile)
    # store into (batch, 0, q, k) — 4D array, tile (1, 1, BM, BN)
    values4 = ct.reshape(values, (1, 1, BM, BN))
    ct.store(out_a, index=(bid_b, 0, bid_i, bid_j), tile=values4)
    ct.store(out_b, index=(bid_b, 0, bid_i, bid_j), tile=values4)


def _materialized_shape(expand_shape):
    return tuple(1 if int(dim) == -1 else int(dim) for dim in expand_shape)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    _source_shape, expand_shape_arg = inputs
    shape = _materialized_shape(expand_shape_arg)
    stride = (shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)
    outs = tuple(
        torch.empty_strided(shape, stride, device="cuda", dtype=torch.bfloat16)
        for _ in range(12)
    )
    BM = 32
    BN = 128
    stream = torch.cuda.current_stream()
    grid = (shape[0], shape[2] // BM, shape[3] // BN)
    # 12 outputs / 2 per kernel launch = 6 launches
    for i in range(0, 12, 2):
        ct.launch(
            stream,
            grid,
            _causal_bias_pair_kernel,
            (outs[i], outs[i + 1], BM, BN, shape[3]),
        )
    return outs
