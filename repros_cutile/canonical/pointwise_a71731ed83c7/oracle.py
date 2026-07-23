"""cuTile port of pointwise_a71731ed83c7: OPT causal source mask.

BANDWIDTH_BOUND: materialize the bool `[4, 1, 2048, 2048]` mask (k<=q) & (arg0[batch,k]!=0).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
SEQ = 2048
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@ct.kernel
def _causal_source_mask_kernel(
    source_ptr,  # f32 [BATCH, SEQ]
    out_ptr,     # bool [BATCH, SEQ, SEQ]
    SEQ_: ct.Constant[int],
    BLOCK_Q: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    batch = ct.bid(0)
    q_block = ct.bid(1)
    k_block = ct.bid(2)

    # Load a (1, BLOCK_K) tile from the source and reshape to (1, 1, BLOCK_K)
    source = ct.load(source_ptr, index=(batch, k_block), shape=(1, BLOCK_K))
    zero = ct.zeros((1, BLOCK_K), dtype=ct.float32)
    source_nonzero = source != zero
    src_bool = ct.reshape(source_nonzero, (1, 1, BLOCK_K))

    # Compute q and k index tiles.
    q_arange = ct.arange(BLOCK_Q, dtype=ct.int32) + q_block * BLOCK_Q
    k_arange = ct.arange(BLOCK_K, dtype=ct.int32) + k_block * BLOCK_K
    q_2d = ct.reshape(q_arange, (BLOCK_Q, 1))
    k_2d = ct.reshape(k_arange, (1, BLOCK_K))
    causal = k_2d <= q_2d  # (BLOCK_Q, BLOCK_K)
    causal_3d = ct.reshape(causal, (1, BLOCK_Q, BLOCK_K))

    values = causal_3d & src_bool  # broadcast (1, 1, BLOCK_K)
    ct.store(out_ptr, index=(batch, q_block, k_block), tile=values)


@oracle_impl(
    hardware="B200",
    point="b8e0cd0a",
    BLOCK_Q=32,
    BLOCK_K=128,
)
def oracle_forward(inputs, *, BLOCK_Q: int, BLOCK_K: int):
    arg0_1, expand_shape = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    out_3d = out.view(BATCH, SEQ, SEQ)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, SEQ // BLOCK_Q, SEQ // BLOCK_K),
        _causal_source_mask_kernel,
        (arg0_1, out_3d, SEQ, BLOCK_Q, BLOCK_K),
    )
    return out.expand(tuple(int(dim) for dim in expand_shape))
