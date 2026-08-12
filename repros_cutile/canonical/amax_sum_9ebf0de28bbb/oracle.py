"""cuTile port of amax_sum_9ebf0de28bbb (NEW_PATTERN): MT5 bidirectional
relative-position attention softmax with an aliased bias tensor as a side
output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS_TOTAL = 32 * 6 * 128
HEADS_C = 6
Q_LEN_C = 128
K_LEN_C = 128


@ct.kernel
def _relative_position_softmax_kernel(
    x_ptr,             # bf16 [ROWS, K]
    rel_bias_ptr,      # bf16 [BUCKETS=32, HEADS]
    bias_out_ptr,      # bf16 flat storage of [B, H, Q, K]
    out_ptr,           # bf16 [ROWS, K]
    ROWS: ct.Constant[int],
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    rows = ct.arange(BLOCK_M, dtype=ct.int64) + row_block * BLOCK_M
    cols = ct.arange(BLOCK_N, dtype=ct.int64)
    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    rows_broad = ct.broadcast_to(rows_2d, (BLOCK_M, BLOCK_N))
    cols_broad = ct.broadcast_to(cols_2d, (BLOCK_M, BLOCK_N))
    row_mask = rows < ROWS
    row_mask_2d = ct.reshape(row_mask, (BLOCK_M, 1))
    col_mask_2d = cols_broad < K_LEN
    row_mask_broad = ct.broadcast_to(row_mask_2d, (BLOCK_M, BLOCK_N))
    active = row_mask_broad & col_mask_2d

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    head = flat_bh - batch * HEADS
    query = rows - flat_bh * Q_LEN

    # Bucket = distance mapping. distance = |cols - query|.
    query_2d = ct.reshape(query, (BLOCK_M, 1))
    query_broad = ct.broadcast_to(query_2d, (BLOCK_M, BLOCK_N))
    rel_pos = cols_broad - query_broad
    distance = ct.where(rel_pos < 0, -rel_pos, rel_pos)

    bucket = distance
    bucket = ct.where(distance >= 8, ct.full((BLOCK_M, BLOCK_N), 8, dtype=ct.int64), bucket)
    bucket = ct.where(distance >= 12, ct.full((BLOCK_M, BLOCK_N), 9, dtype=ct.int64), bucket)
    bucket = ct.where(distance >= 16, ct.full((BLOCK_M, BLOCK_N), 10, dtype=ct.int64), bucket)
    bucket = ct.where(distance >= 23, ct.full((BLOCK_M, BLOCK_N), 11, dtype=ct.int64), bucket)
    bucket = ct.where(distance >= 32, ct.full((BLOCK_M, BLOCK_N), 12, dtype=ct.int64), bucket)
    bucket = ct.where(distance >= 46, ct.full((BLOCK_M, BLOCK_N), 13, dtype=ct.int64), bucket)
    bucket = ct.where(distance >= 64, ct.full((BLOCK_M, BLOCK_N), 14, dtype=ct.int64), bucket)
    bucket = ct.where(distance >= 91, ct.full((BLOCK_M, BLOCK_N), 15, dtype=ct.int64), bucket)
    bucket = bucket + ct.where(rel_pos > 0,
                                ct.full((BLOCK_M, BLOCK_N), 16, dtype=ct.int64),
                                ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.int64))

    # bias offset = bucket * HEADS + head (into rel_bias_ptr flat 1D)
    head_2d = ct.reshape(head, (BLOCK_M, 1))
    head_broad = ct.broadcast_to(head_2d, (BLOCK_M, BLOCK_N))
    bias_lookup = bucket * HEADS + head_broad
    zero64 = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.int64)
    safe_bias = ct.where(active, bias_lookup, zero64)
    bias_bf = ct.gather(rel_bias_ptr, safe_bias)
    # Add 0.0 then cast bf16 -> the +0 does nothing in float; bf16 stays bf16.
    bias_f = ct.astype(bias_bf, ct.float32)
    bias_bf = ct.astype(bias_f, ct.bfloat16)  # round-trip

    # Store bias into strided output layout:
    # bias_offset = batch*(HEADS*Q_LEN*K_LEN) + head + query*(K_LEN*HEADS) + cols*HEADS
    batch_2d = ct.reshape(batch, (BLOCK_M, 1))
    batch_broad = ct.broadcast_to(batch_2d, (BLOCK_M, BLOCK_N))
    query_broad_i = query_broad
    bias_offsets = (
        batch_broad * (HEADS * Q_LEN * K_LEN)
        + head_broad
        + query_broad_i * (K_LEN * HEADS)
        + cols_broad * HEADS
    )
    ct.scatter(bias_out_ptr, bias_offsets, bias_bf, mask=active)

    # Load x, compute softmax.
    row_offsets = rows_broad * K_LEN + cols_broad
    safe_x = ct.where(active, row_offsets, zero64)
    x_bf = ct.gather(x_ptr, safe_x)
    x = ct.astype(x_bf, ct.float32)
    scores = x + ct.astype(bias_bf, ct.float32)
    neg_inf = ct.full((BLOCK_M, BLOCK_N), -1.0e38, dtype=ct.float32)
    scores = ct.where(active, scores, neg_inf)

    row_max = ct.max(scores, axis=1)  # [BLOCK_M]
    row_max_2d = ct.reshape(row_max, (BLOCK_M, 1))
    row_max_2d = ct.where(row_mask_2d, row_max_2d,
                          ct.zeros((BLOCK_M, 1), dtype=ct.float32))
    row_max_broad = ct.broadcast_to(row_max_2d, (BLOCK_M, BLOCK_N))
    numer = ct.exp(scores - row_max_broad)
    numer = ct.where(active, numer, ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    denom_broad = ct.broadcast_to(denom_2d, (BLOCK_M, BLOCK_N))
    probs = numer / denom_broad
    probs_bf = ct.astype(probs, ct.bfloat16)

    ct.scatter(out_ptr, safe_x, probs_bf, mask=active)


@oracle_impl(hardware="B200", point="e7595a1e", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, *_ = inputs
    batch = 32
    heads = HEADS_C
    q_len = Q_LEN_C
    k_len = K_LEN_C
    rows = batch * heads * q_len

    relative_bias = torch.empty_strided(
        (batch, heads, q_len, k_len),
        (heads * q_len * k_len, 1, heads * k_len, heads),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        (batch * heads, q_len, k_len),
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # Flatten x to rows*K
    x_flat = arg0_1.reshape(-1)
    # rel_bias arg1_1 is [BUCKETS, HEADS] (32, 6) contiguous
    rel_bias_flat = arg1_1.reshape(-1)
    out_flat = out.reshape(-1)
    # relative_bias has strided layout; get underlying storage as flat.
    bias_storage_size = int(relative_bias.untyped_storage().nbytes() // relative_bias.element_size())
    bias_flat = relative_bias.as_strided((bias_storage_size,), (1,))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((rows + BLOCK_M - 1) // BLOCK_M, 1, 1),
        _relative_position_softmax_kernel,
        (x_flat, rel_bias_flat, bias_flat, out_flat,
         rows, heads, q_len, k_len, BLOCK_M, BLOCK_N),
    )
    return relative_bias, out
