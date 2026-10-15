"""cuTile port of amax_sum_7bdfc869f56b (NEW_PATTERN): GPT-J scaled additive
attention softmax. For each row: promote x to fp32, divide by 16, add bias
(bf16 promoted to fp32), stable amax/exp/sum/div, then round to bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


K_LEN = 128


@ct.kernel
def _scaled_bias_softmax_kernel(
    x_ptr,      # (batch=16, q=128, K_LEN) f32
    bias_ptr,   # (1, 1, K_LEN, K_LEN) bf16 — broadcast on batch, use (q, K_LEN) slice
    out_ptr,    # (batch, q, K_LEN) bf16
    K: ct.Constant[int],
):
    b = ct.bid(0)
    q = ct.bid(1)
    x = ct.load(x_ptr, index=(b, q, 0), shape=(1, 1, K))
    # bias is [1, 1, 128, 128] with (q, K) slice per row: [0, 0, q, :]
    bias = ct.load(bias_ptr, index=(0, 0, q, 0), shape=(1, 1, 1, K))
    xf = ct.astype(x, ct.float32)
    biasf = ct.astype(bias, ct.float32)
    scores = xf * 0.0625 + ct.reshape(biasf, (1, 1, K))
    row_max = ct.max(scores)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    probs = numer / denom
    ct.store(out_ptr, index=(b, q, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="030f75c1")
def oracle_forward(inputs):
    arg0_1, arg1_1, _s0, _s1, _s2 = inputs
    batch = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    out_shape = (batch, q_len, k_len)
    out = torch.empty_strided(
        out_shape,
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (batch, q_len, 1),
        _scaled_bias_softmax_kernel,
        (arg0_1, arg1_1, out, k_len),
    )
    return out
