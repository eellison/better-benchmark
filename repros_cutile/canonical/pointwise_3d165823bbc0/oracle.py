"""cuTile port of pointwise_3d165823bbc0: PLBart causal + source mask bool [B,1,Q,K]."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _causal_mask_kernel(
    src_mask,   # bool [B, SEQ] (from f32 != 0)
    out_ptr,    # bool [B, SEQ, SEQ]
    SEQ: ct.Constant[int],
    BLOCK_Q: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    b = ct.bid(0)
    q_tile = ct.bid(1)
    k_tile = ct.bid(2)

    # Load src[batch, k_tile*BLOCK_K:(k_tile+1)*BLOCK_K] as [1, BLOCK_K]
    src = ct.load(src_mask, index=(b, k_tile), shape=(1, BLOCK_K))
    src_bool = src != ct.zeros(shape=(1, BLOCK_K), dtype=src.dtype)
    # Broadcast to [BLOCK_Q, BLOCK_K]
    src_2d = ct.full(shape=(BLOCK_Q, BLOCK_K), fill_value=False, dtype=ct.bool_)
    # We'll use where(src_bool, True, False) broadcasted — need a broadcast pattern.
    # cuTile: create a (BLOCK_Q, 1) always-true and multiply by src_bool broadcast
    ones_q = ct.full(shape=(BLOCK_Q, 1), fill_value=True, dtype=ct.bool_)
    # Broadcast (1, BLOCK_K) & (BLOCK_Q, 1) via element-wise and
    src_2d = ones_q & src_bool

    # Causal: k <= q where q_offset = q_tile*BLOCK_Q + arange(BLOCK_Q),
    # k_offset = k_tile*BLOCK_K + arange(BLOCK_K)
    q_range = ct.arange(BLOCK_Q, dtype=ct.int32) + q_tile * BLOCK_Q
    k_range = ct.arange(BLOCK_K, dtype=ct.int32) + k_tile * BLOCK_K
    q_col = ct.reshape(q_range, (BLOCK_Q, 1))
    k_row = ct.reshape(k_range, (1, BLOCK_K))
    causal = k_row <= q_col

    out = causal & src_2d
    out_3d = ct.reshape(out, (1, BLOCK_Q, BLOCK_K))
    ct.store(out_ptr, index=(b, q_tile, k_tile), tile=out_3d)


@oracle_impl(hardware="B200", point="7eeda05f", BLOCK_Q=16, BLOCK_K=128)
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
    # Convert f32 source_mask to bool
    src_bool = (source_mask != 0.0)
    # Reshape base view as [batch, seq, seq]
    base_3d = base.view(batch, seq, seq)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, ct.cdiv(seq, BLOCK_Q), ct.cdiv(seq, BLOCK_K)),
        _causal_mask_kernel,
        (src_bool, base_3d, seq, BLOCK_Q, BLOCK_K),
    )
    return base.expand(tuple(int(dim) for dim in expand_shape))
