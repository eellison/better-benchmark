"""cuTile port of amax_sum_c35490c8001a: MT5 causal relative-position attention softmax.

Ports the Triton `_relative_position_softmax_kernel`: for each row (batch, head,
query) of the [32, 6, 128, 128] attention grid, gather a bf16 bias from the
[32, 6] `rel_bias` table via a log-bucketed causal distance, apply a causal
`-inf` fill for cols > query, add to the bf16 scores, then row-softmax.
Outputs the bias/mask tensor in the head-inner (98304, 1, 768, 6) layout and
the [192, 128, 128] flat softmax probs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH_C = 32
HEADS_C = 6
Q_LEN_C = 128
K_LEN_C = 128
BLOCK_N_C = 128  # equal to K_LEN_C (power of 2)
NEG_LARGE = -3.3895313892515355e38


@ct.kernel
def _relative_position_softmax_kernel(
    x_ptr,             # bf16 [ROWS, K] dense
    rel_bias_ptr,      # bf16 flat [32*6] = 192 (contiguous [32,6])
    bias_out_ptr,      # bf16 flat [B*H*Q*K] with strided layout
    out_ptr,           # bf16 [ROWS, K] dense
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bh = row // Q_LEN
    batch = flat_bh // HEADS
    head = flat_bh - batch * HEADS
    query = row - flat_bh * Q_LEN

    cols = ct.arange(BLOCK_N, dtype=ct.int64)
    query_i64 = ct.astype(query, ct.int64)
    diff = query_i64 - cols
    zero_i64 = ct.zeros((BLOCK_N,), dtype=ct.int64)
    neg = ct.where(diff > 0, diff, zero_i64)

    # T5-style bucket: for distance < 16 use distance directly;
    # otherwise use min(16 + floor(log(distance/16)/log(8) * 16), 31).
    lt = neg < 16
    neg_f = ct.astype(neg, ct.float32)
    log_arg_raw = neg_f * (1.0 / 16.0)
    one_f = ct.full((BLOCK_N,), 1.0, dtype=ct.float32)
    log_input = ct.where(log_arg_raw > 0.0, log_arg_raw, one_f)
    log_val = ct.log(log_input)
    div_val = log_val * (1.0 / 2.0794415416798357)
    mul_val = div_val * 16.0
    idx = ct.astype(mul_val, ct.int64) + 16
    cap = ct.full((BLOCK_N,), 31, dtype=ct.int64)
    bucket_large = ct.where(idx > 31, cap, idx)
    bucket = ct.where(lt, neg, bucket_large)

    # Gather bias from rel_bias[bucket, head]. Flat index = bucket * HEADS + head.
    bias_idx_1d = bucket * HEADS + ct.astype(head, ct.int64)
    bias_bf = ct.gather(rel_bias_ptr, bias_idx_1d)

    # Causal mask: cols <= query. Non-causal cols get replaced with -3.39e+38.
    causal = cols <= query_i64
    zero_f = ct.zeros((BLOCK_N,), dtype=ct.float32)
    neg_large_f = ct.full((BLOCK_N,), NEG_LARGE, dtype=ct.float32)
    fill = ct.where(causal, zero_f, neg_large_f)
    bias_mask_f = ct.astype(bias_bf, ct.float32) + fill
    bias_mask_bf = ct.astype(bias_mask_f, ct.bfloat16)

    # bias_out has strides (HEADS*Q*K, 1, K*HEADS, HEADS) — head-inner layout.
    # flat[b, h, q, k] = b*(H*Q*K) + h + q*(K*H) + k*H
    batch_i64 = ct.astype(batch, ct.int64)
    head_i64 = ct.astype(head, ct.int64)
    bias_offsets = (
        batch_i64 * (HEADS * Q_LEN * K_LEN)
        + head_i64
        + query_i64 * (K_LEN * HEADS)
        + cols * HEADS
    )
    ct.scatter(bias_out_ptr, bias_offsets, bias_mask_bf)

    # Row softmax: scores = x + bias_mask (roundtrip through bf16 for bias_mask).
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    x_1d = ct.reshape(x, (BLOCK_N,))
    x_f = ct.astype(x_1d, ct.float32)
    scores = x_f + ct.astype(bias_mask_bf, ct.float32)

    row_max = ct.max(scores)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    probs_2d = ct.reshape(probs_bf, (1, BLOCK_N))
    ct.store(out_ptr, index=(row, 0), tile=probs_2d)


@oracle_impl(hardware="B200", point="e7595a1e", BLOCK_N=BLOCK_N_C)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, *_ = inputs
    device = arg0_1.device
    batch = BATCH_C
    heads = HEADS_C
    q_len = Q_LEN_C
    k_len = K_LEN_C
    rows = batch * heads * q_len

    # arg0_1 is bf16 [192, 128, 128] (contiguous); view as [rows, K].
    x_2d = arg0_1.contiguous().view(rows, k_len)

    # arg1_1 is bf16 [32, 6] contiguous.
    rel_bias_flat = arg1_1.contiguous().view(-1)

    # bias output: [32, 6, 128, 128] with head-inner strides.
    bias = torch.empty_strided(
        (batch, heads, q_len, k_len),
        (heads * q_len * k_len, 1, heads * k_len, heads),
        device=device, dtype=torch.bfloat16,
    )
    # Access bias's flat storage via as_strided.
    bias_storage_size = int(bias.untyped_storage().nbytes() // bias.element_size())
    bias_flat = bias.as_strided((bias_storage_size,), (1,))

    # softmax probs output: [192, 128, 128] contiguous.
    out = torch.empty_strided(
        (batch * heads, q_len, k_len),
        (q_len * k_len, k_len, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_2d = out.view(rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _relative_position_softmax_kernel,
        (x_2d, rel_bias_flat, bias_flat, out_2d, heads, q_len, k_len, BLOCK_N),
    )
    return bias, out
