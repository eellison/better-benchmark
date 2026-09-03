"""cuTile port of amax_sum_02064a1e60ac: TrOCR broadcast-bias attention softmax."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_HEADS = 16
Q_LEN = 256
K_LEN = 256
BATCHES = 64
BLOCK_H = 8


@ct.kernel
def _broadcast_add_softmax_kernel(
    x_ptr,          # bf16 [batch, heads, q_len, k_len]
    bias_ptr,       # f32  [batch, 1, q_len, k_len]
    amax_ptr,       # f32  [batch, heads, q_len, 1]
    sum_ptr,        # f32  [batch, heads, q_len, 1]
    out_ptr,        # bf16 [batch, heads, q_len, k_len]
    K_LEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
):
    b = ct.bid(0)
    h_block = ct.bid(1)
    q = ct.bid(2)

    x = ct.load(x_ptr, index=(b, h_block, q, 0),
                shape=(1, BLOCK_H_C, 1, K_LEN_C))
    bias = ct.load(bias_ptr, index=(b, 0, q, 0), shape=(1, 1, 1, K_LEN_C))
    scores2d = ct.reshape(ct.astype(x, ct.float32), (BLOCK_H_C, K_LEN_C))
    bias2d = ct.reshape(bias, (1, K_LEN_C))
    scores = scores2d + bias2d

    row_max = ct.max(scores, axis=1, keepdims=True)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom

    amax_4d = ct.reshape(row_max, (1, BLOCK_H_C, 1, 1))
    sum_4d = ct.reshape(denom, (1, BLOCK_H_C, 1, 1))
    out_4d = ct.reshape(ct.astype(probs, ct.bfloat16), (1, BLOCK_H_C, 1, K_LEN_C))
    ct.store(amax_ptr, index=(b, h_block, q, 0), tile=amax_4d)
    ct.store(sum_ptr, index=(b, h_block, q, 0), tile=sum_4d)
    ct.store(out_ptr, index=(b, h_block, q, 0), tile=out_4d)


@oracle_impl(hardware="B200", point="19ef62d0")
def oracle_forward(inputs):
    arg0_1, arg1_1, view_shape_arg, out_shape_arg = inputs
    x4 = arg0_1.view(BATCHES, N_HEADS, Q_LEN, K_LEN)
    device = arg0_1.device
    amax = torch.empty_strided(
        (BATCHES, N_HEADS, Q_LEN, 1),
        (N_HEADS * Q_LEN, Q_LEN, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        (BATCHES, N_HEADS, Q_LEN, 1),
        (N_HEADS * Q_LEN, Q_LEN, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (1024, 256, 256),
        (256 * 256, 256, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out4 = out.view(BATCHES, N_HEADS, Q_LEN, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCHES, N_HEADS // BLOCK_H, Q_LEN),
        _broadcast_add_softmax_kernel,
        (x4, arg1_1, amax, sum_1, out4, K_LEN, BLOCK_H),
    )
    amax_view = amax.view(1024, 256, 1)
    sum_view = sum_1.view(1024, 256, 1)
    return amax_view, sum_view, out, out.permute(0, 2, 1)
