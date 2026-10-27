"""cuTile port of amax_sum_4506a84dfa49: BERT masked scaled attention softmax (heads=12, q_len=k_len=128)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
HEADS = 12
Q_LEN = 128
K_LEN = 128


@ct.kernel
def _masked_scaled_softmax_kernel(
    mask_ptr,   # (batch, 1, q_len, k_len) bool
    scores_ptr, # (batch*heads, q_len, k_len) bf16
    out_ptr,    # (batch*heads, q_len, k_len) bf16
    BLOCK_H: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    # Grid: (batch, q_len, heads // BLOCK_H)
    batch = ct.bid(0)
    q = ct.bid(1)
    head_tile = ct.bid(2)

    # Load mask: (batch, 1, q, :K_LEN) => shape (1, 1, 1, BLOCK_K)
    keep = ct.load(mask_ptr, index=(batch, 0, q, 0), shape=(1, 1, 1, BLOCK_K))
    keep_flat = ct.reshape(keep, (BLOCK_K,))
    # Load scores tile: (batch_head, q, :) with batch_head range
    # We use the (batch*heads, q_len, k_len) view; batch_head_start = batch*heads + head_tile*BLOCK_H
    # cuTile can't multiply grid indices, so pass a 4D view (batch, heads, q_len, k_len)
    scores = ct.load(scores_ptr, index=(batch, head_tile, q, 0), shape=(1, BLOCK_H, 1, BLOCK_K))
    scores_2d = ct.reshape(scores, (BLOCK_H, BLOCK_K))
    scores_f = ct.astype(scores_2d, ct.float32)
    # keep_flat broadcast to (BLOCK_H, BLOCK_K)
    keep_2d = ct.reshape(keep_flat, (1, BLOCK_K))

    fill = ct.full(shape=(BLOCK_H, BLOCK_K), fill_value=-998244352.0, dtype=ct.float32)
    scaled = scores_f * 0.125
    # keep_2d is bool broadcast
    x = ct.where(keep_2d, scaled, fill)

    row_max = ct.max(x, axis=1, keepdims=True)
    numer = ct.exp(x - row_max)
    # Zero out invalid columns
    zero = ct.zeros(shape=(BLOCK_H, BLOCK_K), dtype=ct.float32)
    numer = ct.where(keep_2d, numer, zero)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    probs_4d = ct.reshape(probs_bf, (1, BLOCK_H, 1, BLOCK_K))
    ct.store(out_ptr, index=(batch, head_tile, q, 0), tile=probs_4d)


@oracle_impl(hardware="B200", point="b762f7d9", BLOCK_H=4, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_H, BLOCK_K):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    batch = int(arg0_1.shape[0])
    heads = int(_shape_param_2[0]) // batch
    q_len = int(arg1_1.shape[1])
    k_len = int(arg1_1.shape[2])
    assert heads == HEADS
    assert q_len == Q_LEN
    assert k_len == K_LEN
    assert heads % BLOCK_H == 0, f"heads={heads} not divisible by BLOCK_H={BLOCK_H}"
    assert k_len == BLOCK_K

    # View scores/out as (batch, heads, q_len, k_len).
    scores_4d = arg1_1.view(batch, heads, q_len, k_len)
    out_4d = out.view(batch, heads, q_len, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, q_len, heads // BLOCK_H),
        _masked_scaled_softmax_kernel,
        (arg0_1, scores_4d, out_4d, BLOCK_H, BLOCK_K),
    )
    return out
