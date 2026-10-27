"""cuTile port of amax_sum_9675d6f1ba21: GPT-J scaled+biased attention
softmax with sibling f32/bf16 outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scaled_biased_softmax_kernel(
    x_ptr,          # bf16 (heads, q_len, k_len)
    bias_ptr,       # f32 (q_len, k_len)
    out_f32_ptr,    # f32 (heads, q_len, k_len)
    out_bf16_ptr,   # bf16 (heads, q_len, k_len)
    K_LEN: ct.Constant[int],
):
    head = ct.bid(0)
    q = ct.bid(1)
    x = ct.load(x_ptr, index=(head, q, 0), shape=(1, 1, K_LEN))
    bias = ct.load(bias_ptr, index=(q, 0), shape=(1, K_LEN))
    x_f = ct.astype(x, ct.float32)
    bias_2d = ct.reshape(bias, (1, 1, K_LEN))
    scores = x_f * 0.0625 + ct.astype(bias_2d, ct.float32)

    row_max = ct.max(scores)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer)
    probs = numer / denom

    ct.store(out_f32_ptr, index=(head, q, 0), tile=probs)
    ct.store(out_bf16_ptr, index=(head, q, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="c2b7b80f", BLOCK_M=4, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    heads = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])

    # arg1_1 is f32 (1, 1, q_len, k_len); view as (q_len, k_len) for broadcast.
    bias_view = arg1_1.view(q_len, k_len)

    out_f32 = torch.empty_strided(
        (1, heads, q_len, k_len),
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (heads, q_len, 1),
        _scaled_biased_softmax_kernel,
        (arg0_1, bias_view, out_f32.view(heads, q_len, k_len), out_bf16, k_len),
    )

    return out_f32, out_bf16, out_bf16.permute(0, 2, 1)
