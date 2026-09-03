"""cuTile port of amax_sum_13a9aaa2eadb: GPT-Neo masked bf16 scalar-fill softmax over K=128.

View scores as [BATCH, HEADS, Q_LEN, K_LEN]. Tile with (1, BLOCK_H, 1, BLOCK_K).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HEADS = 16
Q_LEN = 128
K_LEN = 128
BLOCK_K = 128
BLOCK_H = 4
FILL_MIN = -3.4028234663852886e38


@ct.kernel
def _masked_scalar_bias_softmax_kernel(
    mask_ptr,   # b8 [Q_LEN, K_LEN]
    scores_ptr, # f32 [BATCH, HEADS, Q_LEN, K_LEN]
    bias_ptr,   # bf16 [BATCH, Q_LEN, K_LEN]
    out_ptr,    # bf16 [BATCH, HEADS, Q_LEN, K_LEN]
    K_LEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
    BLOCK_K_C: ct.Constant[int],
):
    batch = ct.bid(0)
    head_block = ct.bid(1)
    q = ct.bid(2)

    keep = ct.load(mask_ptr, index=(q, 0), shape=(1, BLOCK_K_C))
    bias = ct.load(bias_ptr, index=(batch, q, 0), shape=(1, 1, BLOCK_K_C))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_K_C))

    # scores tile-space: (batch, head_block, q, 0) with shape (1, BLOCK_H, 1, BLOCK_K)
    scores = ct.load(scores_ptr, index=(batch, head_block, q, 0),
                     shape=(1, BLOCK_H_C, 1, BLOCK_K_C))
    scores_2d = ct.reshape(scores, (BLOCK_H_C, BLOCK_K_C))

    fill = ct.full(shape=(BLOCK_H_C, BLOCK_K_C), fill_value=FILL_MIN,
                   dtype=ct.float32)
    keep_2d = ct.reshape(keep, (1, BLOCK_K_C))
    x = ct.where(keep_2d, scores_2d, fill) + bias_2d
    row_max = ct.max(x, axis=1, keepdims=True)
    numer = ct.exp(x - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    out = ct.astype(probs, ct.bfloat16)
    out_4d = ct.reshape(out, (1, BLOCK_H_C, 1, BLOCK_K_C))
    ct.store(out_ptr, index=(batch, head_block, q, 0), tile=out_4d)


@oracle_impl(hardware="B200", point="0b7018c4")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, _shape0, _shape1, _shape2 = inputs
    batch = int(arg2_1.shape[0])

    # Reshape mask to [Q_LEN, K_LEN] — no contiguous needed if strides are already
    # compatible. arg0_1 shape [1,1,2048,2048] contiguous → viewing top-left
    # [128,128] slice is a view, but load needs a tensor of shape [Q_LEN, K_LEN]
    # sub-array. Simplest: slice with .contiguous once (small buffer).
    mask_2d = arg0_1.view(2048, 2048)[:Q_LEN, :K_LEN].contiguous()

    # Scores as [BATCH, HEADS, Q_LEN, K_LEN]
    scores_4d = arg1_1.view(batch, HEADS, Q_LEN, K_LEN)
    bias_3d = arg2_1.view(batch, Q_LEN, K_LEN)

    out = torch.empty_strided(
        (batch * HEADS, Q_LEN, K_LEN),
        (Q_LEN * K_LEN, K_LEN, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out_4d = out.view(batch, HEADS, Q_LEN, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, HEADS // BLOCK_H, Q_LEN),
        _masked_scalar_bias_softmax_kernel,
        (mask_2d, scores_4d, bias_3d, out_4d, K_LEN, BLOCK_H, BLOCK_K),
    )
    return out
