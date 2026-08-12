"""cuTile port of amax_sum_57fd7ff76261: T5 causal relative-position attention softmax.

Strategy: the T5 causal relative-position bucket build + embedding lookup +
causal mask is expressed in torch (matches the returned `add5` tensor
element-for-element). The cuTile kernel then runs the fused row-softmax over
`scores + add5` where add5 is broadcast across batches — matching the
[64,1024,1024] permute path of the Triton oracle.

Returns (add5, probs).
"""

import math

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


MASK_VALUE = -3.3895313892515355e38


def _t5_causal_bucket(distance, num_buckets, max_distance):
    """Match transformers' T5 relative_position_bucket for causal (bidirectional=False).

    distance: i64[q,k] with distance = max(query - key, 0) (already positive).
    """
    max_exact = num_buckets // 2  # 16
    is_small = distance < max_exact
    log_ratio = torch.log(distance.float() / max_exact) / math.log(max_distance / max_exact)
    val_if_large = max_exact + (log_ratio * max_exact).to(torch.int64)
    val_if_large = torch.minimum(
        val_if_large,
        torch.full_like(val_if_large, num_buckets - 1),
    )
    return torch.where(is_small, distance, val_if_large)


def _build_add5(rel_bias, batch, heads, q_len, k_len, device):
    """Materialize the [batch, heads, q_len, k_len] bf16 add5 tensor."""
    q_idx = torch.arange(q_len, device=device).unsqueeze(-1)  # [q, 1]
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)   # [1, k]
    rel = k_idx - q_idx                                       # [q, k]
    causal = rel <= 0
    distance = torch.maximum(-rel, torch.zeros_like(rel))     # [q, k]
    bucket = _t5_causal_bucket(distance, num_buckets=32, max_distance=128)  # [q, k]

    embed = rel_bias[bucket]           # [q, k, heads] bf16
    rel_add = embed.permute(2, 0, 1)   # [heads, q, k] bf16

    # add5 = broadcast rel_add + causal_mask across batch, heads with MASK_VALUE
    # outside causal positions.
    mask_value = torch.full((), MASK_VALUE, dtype=torch.bfloat16, device=device)
    add5_head = torch.where(causal, rel_add, mask_value)          # [heads, q, k]
    add5 = add5_head.unsqueeze(0).expand(batch, heads, q_len, k_len).contiguous()
    return add5


@ct.kernel
def _softmax_with_bias_kernel(
    scores_ptr,     # bf16 [n_rows, K]
    bias_ptr,       # bf16 [BATCH, HEADS, Q, K]
    probs_ptr,      # bf16 [n_rows, K]
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bh = row // Q_LEN
    batch = flat_bh // HEADS
    head = flat_bh - (flat_bh // HEADS) * HEADS
    query = row - flat_bh * Q_LEN

    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_K))
    bias_bf = ct.load(bias_ptr, index=(batch, head, query, 0), shape=(1, 1, 1, BLOCK_K))
    bias_2d = ct.reshape(bias_bf, (1, BLOCK_K))

    scores_f = ct.astype(scores_bf, ct.float32)
    logits = scores_f + ct.astype(bias_2d, ct.float32)

    row_max = ct.max(logits, axis=1, keepdims=True)
    numer = ct.exp(logits - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)
    ct.store(probs_ptr, index=(row, 0), tile=probs)


@oracle_impl(hardware="B200", point="ea4c0a34", BLOCK_K=1024)
def oracle_forward(inputs, *, BLOCK_K: int):
    (
        arg0_1, arg1_1,
        _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3,
        _shape_param_4, _shape_param_5,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    n_heads = int(arg1_1.shape[1])
    n_batch = int(arg0_1.shape[0]) // n_heads
    n_rows = n_batch * n_heads * q_len
    device = arg0_1.device

    add5 = _build_add5(arg1_1, n_batch, n_heads, q_len, k_len, device)
    # Match the exact stride the Triton oracle uses.
    add5_shape = tuple(int(dim) for dim in _shape_param_4)  # (8, 8, 1024, 1024)
    add5_target = torch.empty_strided(
        add5_shape,
        (n_heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=device, dtype=torch.bfloat16,
    )
    add5_target.copy_(add5)

    probs_shape = tuple(int(dim) for dim in _shape_param_5)  # (64, 1024, 1024)
    probs = torch.empty_strided(
        probs_shape,
        tuple(int(stride) for stride in arg0_1.stride()),
        device=device, dtype=torch.bfloat16,
    )

    scores_2d = arg0_1.contiguous().view(n_rows, k_len)
    probs_2d = probs.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_with_bias_kernel,
        (scores_2d, add5_target, probs_2d, n_heads, q_len, k_len, BLOCK_K),
    )
    return add5_target, probs
