"""cuTile port of pointwise_f3d3e860d8b0: Pegasus causal source-mask."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
SEQ = 128
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@ct.kernel
def _causal_source_mask_kernel(
    in_ptr,        # f32 [BATCH, SEQ]
    out_ptr,       # bool [BATCH, 1, SEQ, SEQ]
    BLOCK_Q: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    b = ct.bid(0)
    q_tile = ct.bid(1)
    k_tile = ct.bid(2)

    # Load source column [BLOCK_K] slice
    source = ct.load(in_ptr, index=(b, k_tile), shape=(1, BLOCK_K))
    source_mask = source != 0.0
    # Reshape source_mask to (1, BLOCK_K) already
    source_mask_row = ct.reshape(source_mask, (1, BLOCK_K))

    # Build causal predicate: cols <= rows
    q_start = q_tile * BLOCK_Q
    k_start = k_tile * BLOCK_K
    q_idx = q_start + ct.arange(BLOCK_Q, dtype=ct.int32)
    k_idx = k_start + ct.arange(BLOCK_K, dtype=ct.int32)
    q_2d = ct.reshape(q_idx, (BLOCK_Q, 1))
    k_2d = ct.reshape(k_idx, (1, BLOCK_K))
    causal = k_2d <= q_2d  # (BLOCK_Q, BLOCK_K)
    # Broadcast source_mask (1, BLOCK_K) against causal (BLOCK_Q, BLOCK_K)
    combined = causal & source_mask_row
    tile4d = ct.reshape(combined, (1, 1, BLOCK_Q, BLOCK_K))
    ct.store(out_ptr, index=(b, 0, q_tile, k_tile), tile=tile4d)


@oracle_impl(hardware="B200", point="60ceae96", BLOCK_Q=32, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_Q: int, BLOCK_K: int):
    arg0_1, _shape_param_0 = inputs
    del _shape_param_0
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, ct.cdiv(SEQ, BLOCK_Q), ct.cdiv(SEQ, BLOCK_K)),
        _causal_source_mask_kernel,
        (arg0_1, out, BLOCK_Q, BLOCK_K),
    )
    return out
