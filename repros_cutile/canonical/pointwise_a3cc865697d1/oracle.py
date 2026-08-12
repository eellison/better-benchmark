"""cuTile port of pointwise_a3cc865697d1: GPT-2 segment causal attention mask.

Builds an iota side output plus a bf16 [B,1,SEQ,SEQ] mask using positional
and segment lookups. Uses ct.gather for the segment table row lookups.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _iota_kernel(iota_ptr, BATCH: ct.Constant[int]):
    tile = ct.arange(BATCH, dtype=ct.int64)
    ct.store(iota_ptr, index=(0,), tile=tile)


@ct.kernel
def _segment_mask_kernel(
    position_ptr,   # int64 [SEQ]  (row-broadcast over batch)
    segment_ptr,    # int64 [BATCH, SEQ]
    out_ptr,        # bf16 [BATCH, SEQ, SEQ]
    SEQ: ct.Constant[int],
    BLOCK_Q: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    batch = ct.bid(0)
    q_block = ct.bid(1)
    k_block = ct.bid(2)

    q_pos = ct.load(position_ptr, index=(q_block,), shape=(BLOCK_Q,))
    k_pos = ct.load(position_ptr, index=(k_block,), shape=(BLOCK_K,))

    # Gather segments: segment[batch, q_pos] and segment[batch, k_pos]
    batch_q = ct.full(shape=(BLOCK_Q,), fill_value=batch, dtype=ct.int64)
    batch_k = ct.full(shape=(BLOCK_K,), fill_value=batch, dtype=ct.int64)
    q_segment = ct.gather(segment_ptr, (batch_q, q_pos))
    k_segment = ct.gather(segment_ptr, (batch_k, k_pos))

    q_pos_2d = ct.reshape(q_pos, (BLOCK_Q, 1))
    k_pos_2d = ct.reshape(k_pos, (1, BLOCK_K))
    q_seg_2d = ct.reshape(q_segment, (BLOCK_Q, 1))
    k_seg_2d = ct.reshape(k_segment, (1, BLOCK_K))

    keep_causal = k_pos_2d <= q_pos_2d
    keep_segment = k_seg_2d == q_seg_2d
    keep = keep_causal & keep_segment

    zero = ct.full(shape=(BLOCK_Q, BLOCK_K), fill_value=0.0, dtype=ct.float32)
    neg_inf = ct.full(shape=(BLOCK_Q, BLOCK_K), fill_value=float("-inf"), dtype=ct.float32)
    values = ct.astype(ct.where(keep, zero, neg_inf), ct.bfloat16)
    # Expand to (1, BLOCK_Q, BLOCK_K) to match the 3D output partitioning.
    values_3d = ct.reshape(values, (1, BLOCK_Q, BLOCK_K))
    ct.store(out_ptr, index=(batch, q_block, k_block), tile=values_3d)


@oracle_impl(hardware="B200", point="1536bfa0", BLOCK_Q=16, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_Q: int, BLOCK_K: int):
    positions, segments, _shape_param_0, shape_param_1 = inputs
    del _shape_param_0

    batch = int(segments.shape[0])
    seq = int(positions.shape[1])

    iota = torch.empty_strided((batch,), (1,), device=positions.device, dtype=torch.int64)
    base = torch.empty_strided(
        (batch, 1, seq, seq),
        (seq * seq, seq * seq, seq, 1),
        device=positions.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    # iota output
    ct.launch(stream, (1, 1, 1), _iota_kernel, (iota, batch))

    # Positions is (1, seq); flatten to (seq,)
    positions_1d = positions.view(seq)
    base_3d = base.view(batch, seq, seq)

    grid = (batch, ct.cdiv(seq, BLOCK_Q), ct.cdiv(seq, BLOCK_K))
    ct.launch(
        stream,
        grid,
        _segment_mask_kernel,
        (positions_1d, segments, base_3d, seq, BLOCK_Q, BLOCK_K),
    )
    return iota, base, base.expand(tuple(int(dim) for dim in shape_param_1))
