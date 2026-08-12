"""cuTile port of amax_sum_ccbdc7f5ab12: GPT-J causal same-segment masked softmax.

Produces two outputs:
1. `where`: bf16 [1,1,128,128] mask tensor with keep=0, drop=-3.389e38 (bf16 min).
2. `probs`: bf16 [1,16,128,128] softmax(scores/16 + mask).

Two kernels:
- Mask kernel: builds bf16 [q_len, k_len] where tensor once.
- Softmax kernel: per (head, query), loads mask + scores, computes softmax.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BF16_MIN = -3.3895313892515355e38


@ct.kernel
def _mask_kernel(
    segments_ptr,   # i64 [q_len]
    where_ptr,      # bf16 [q_len, k_len]
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BF16_MIN_: ct.Constant[float],
):
    query = ct.bid(0)
    q_seg = ct.load(segments_ptr, index=(query,), shape=(1,))
    k_seg = ct.load(segments_ptr, index=(0,), shape=(K_LEN,))

    cols = ct.arange(K_LEN, dtype=ct.int32)
    q_seg_bcast = ct.reshape(q_seg, (1,))
    keep = (cols <= query) & (k_seg == q_seg_bcast)

    where_val = ct.where(
        keep,
        ct.zeros((K_LEN,), dtype=ct.bfloat16),
        ct.full(shape=(K_LEN,), fill_value=BF16_MIN_, dtype=ct.bfloat16),
    )
    where_2d = ct.reshape(where_val, (1, K_LEN))
    ct.store(where_ptr, index=(query, 0), tile=where_2d)


@ct.kernel
def _softmax_kernel(
    scores_ptr,   # f32 [heads, q_len, k_len]
    where_ptr,    # bf16 [q_len, k_len]
    out_ptr,      # bf16 [heads, q_len, k_len]
    K_LEN: ct.Constant[int],
):
    head = ct.bid(0)
    query = ct.bid(1)

    scores = ct.load(scores_ptr, index=(head, query, 0), shape=(1, 1, K_LEN))
    where_2d = ct.load(where_ptr, index=(query, 0), shape=(1, K_LEN))
    where_3d = ct.reshape(where_2d, (1, 1, K_LEN))
    where_f = ct.astype(where_3d, ct.float32)
    masked = scores * 0.0625 + where_f

    # softmax
    row_max = ct.max(masked)
    numer = ct.exp(masked - row_max)
    denom = ct.sum(numer)
    probs = numer / denom
    ct.store(out_ptr, index=(head, query, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="94a9fea0")
def oracle_forward(inputs):
    arg0_1, arg1_1, _s0, _s1, _s2, _s3 = inputs
    heads = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])

    where_out = torch.empty_strided(
        (1, 1, q_len, k_len),
        (q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    probs_out = torch.empty_strided(
        (heads, q_len, k_len),
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    where_2d = where_out.view(q_len, k_len)
    segments_1d = arg1_1.view(-1).to(torch.int64)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (q_len, 1, 1),
        _mask_kernel,
        (segments_1d, where_2d, q_len, k_len, BF16_MIN),
    )
    ct.launch(
        stream,
        (heads, q_len, 1),
        _softmax_kernel,
        (arg0_1, where_2d, probs_out, k_len),
    )
    return where_out, probs_out
