"""cuTile port of amax_sum_68ce17870caf: BERT bf16 token-mask scaled softmax.

Matches Triton's single-kernel structure: one kernel produces both the bool
token mask side output and the bf16 softmax probabilities, using
BLOCK_H=8 heads and BLOCK_K=128 keys. Grid = (batch*q_len, cdiv(heads, BLOCK_H)).
Only the head_block==0 program writes the mask (matches Triton's guarded store).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BF16_FILL = -998244352.0


@ct.kernel
def _token_masked_scaled_softmax_kernel(
    token_ptr,     # i64 (batch, k_len) flat
    scores_ptr,    # bf16 (batch*heads, q_len, k_len) flat
    mask_out_ptr,  # bool (batch, 1, q_len, k_len) flat
    out_ptr,       # bf16 (batch*heads, q_len, k_len) flat
    scores_s0: ct.Constant[int],
    scores_s1: ct.Constant[int],
    scores_s2: ct.Constant[int],
    out_s0: ct.Constant[int],
    out_s1: ct.Constant[int],
    out_s2: ct.Constant[int],
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    FILL: ct.Constant[float],
):
    batch_q = ct.bid(0)
    head_block = ct.bid(1)

    batch = batch_q // Q_LEN
    q = batch_q - batch * Q_LEN
    head_offsets = head_block * BLOCK_H + ct.arange(BLOCK_H, dtype=ct.int32)
    cols = ct.arange(BLOCK_K, dtype=ct.int32)

    head_mask = head_offsets < HEADS
    col_mask = cols < K_LEN
    elem_mask = ct.reshape(head_mask, (BLOCK_H, 1)) & ct.reshape(col_mask, (1, BLOCK_K))

    # Token ids for this batch row (length K_LEN).
    token_offsets = batch * K_LEN + cols
    ids = ct.gather(token_ptr, token_offsets, mask=col_mask, padding_value=0)
    zeros = ct.zeros(shape=(BLOCK_K,), dtype=ct.int64)
    keep = ids > zeros

    # Store mask output only from head_block == 0 to avoid duplicate writes.
    mask_offsets = batch * (Q_LEN * K_LEN) + q * K_LEN + cols
    ct.scatter(mask_out_ptr, mask_offsets, keep,
               mask=col_mask & (head_block == 0))

    # Load scores for this (batch, head_block heads, q, all keys).
    flat_heads = batch * HEADS + head_offsets  # (BLOCK_H,)
    flat_heads_2d = ct.reshape(flat_heads, (BLOCK_H, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_K))
    score_offsets = (
        flat_heads_2d * scores_s0
        + q * scores_s1
        + cols_2d * scores_s2
    )
    scores = ct.gather(scores_ptr, score_offsets, mask=elem_mask, padding_value=0.0)
    scores_f = ct.astype(scores, ct.float32)

    keep_2d = ct.reshape(keep, (1, BLOCK_K))
    fill_2d = ct.full(shape=(1, 1), fill_value=FILL, dtype=ct.float32)
    scaled = scores_f * 0.125
    x = ct.where(keep_2d & elem_mask, scaled, fill_2d)

    row_max = ct.max(x, axis=1)
    row_max_2d = ct.reshape(row_max, (BLOCK_H, 1))
    numer = ct.exp(x - row_max_2d)
    numer_masked = ct.where(elem_mask, numer, ct.full(shape=(1, 1), fill_value=0.0, dtype=ct.float32))
    denom = ct.sum(numer_masked, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_H, 1))
    probs = numer_masked / denom_2d

    out_offsets = (
        flat_heads_2d * out_s0
        + q * out_s1
        + cols_2d * out_s2
    )
    ct.scatter(out_ptr, out_offsets, ct.astype(probs, ct.bfloat16), mask=elem_mask)


@oracle_impl(hardware="B200", point="1972aff7", BLOCK_H=8, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_H: int, BLOCK_K: int):
    arg0_1, arg1_1, _sp0, _sp1, _sp2, sp3 = inputs
    out_shape = tuple(int(d) for d in sp3)

    mask_out = torch.empty_strided(
        (16, 1, 128, 128),
        (16384, 16384, 128, 1),
        device=arg1_1.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    batch = int(arg0_1.shape[0])
    q_len = int(arg1_1.shape[1])
    k_len = int(arg1_1.shape[2])
    heads = int(out_shape[0]) // batch

    # Flat views for scatter/gather (metadata-only).
    token_flat = arg0_1.view(-1)
    scores_flat = arg1_1.view(-1)
    mask_flat = mask_out.view(-1)
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch * q_len, ct.cdiv(heads, BLOCK_H), 1),
        _token_masked_scaled_softmax_kernel,
        (token_flat, scores_flat, mask_flat, out_flat,
         int(arg1_1.stride(0)), int(arg1_1.stride(1)), int(arg1_1.stride(2)),
         int(out.stride(0)), int(out.stride(1)), int(out.stride(2)),
         heads, q_len, k_len, BLOCK_H, BLOCK_K, BF16_FILL),
    )
    return mask_out, out
