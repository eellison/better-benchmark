"""cuTile port of amax_sum_821fb95bd167: Swin relative-position softmax with pad.

Kernel-1: materialize relpos bias table from index+table. Kernel-2: for each
row (of 56 total per score row, but only rows [0:49] are active), compute
softmax across cols[0:49], store bf16 probs to cols[0:49] of a 56-wide row.
Uses BLOCK_N=64 with col_store<56 masking to satisfy cuTile store granularity.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_N = 64
BLOCK_BIAS = 256


@ct.kernel
def _materialize_relpos_bias_kernel(
    index_ptr,   # i64 (2401,)
    table_ptr,   # bf16 (169, HEADS,)
    bias_ptr,    # bf16 (HEADS * 2401,)
    HEADS: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    mask = offs < TOTAL
    head = offs // 2401
    rel = offs - head * 2401
    rel_index_i64 = ct.gather(index_ptr, rel, mask=mask, padding_value=0)
    rel_index = ct.astype(rel_index_i64, ct.int32)
    bias = ct.gather(table_ptr, rel_index * HEADS + head, mask=mask, padding_value=0.0)
    ct.scatter(bias_ptr, offs, bias, mask=mask)


@ct.kernel
def _swin_softmax_kernel(
    scores_ptr,   # bf16 (N_SCORE_ROWS * 3136,) — scores has shape [*,56,56]
    bias_ptr,     # bf16 (HEADS * 2401,)
    out_ptr,      # bf16 (N_ROWS, 56) — flat, N_ROWS = N_SCORE_ROWS * 56
    N_ROWS: ct.Constant[int],
    HEADS: ct.Constant[int],
    BLOCK_N_C: ct.Constant[int],
):
    row = ct.bid(0)  # row in [0, N_ROWS)
    cols = ct.arange(BLOCK_N_C, dtype=ct.int32)

    score_row = row // 56
    query = row - score_row * 56
    head = score_row - (score_row // HEADS) * HEADS
    row_live = query < 49
    col_live = cols < 49
    active = col_live  # per-row mask handled outside via row_live

    score_offs = score_row * 3136 + query * 56 + cols
    bias_offs = head * 2401 + query * 49 + cols

    score = ct.astype(
        ct.gather(scores_ptr, score_offs, mask=active, padding_value=0),
        ct.float32)
    bias = ct.astype(
        ct.gather(bias_ptr, bias_offs, mask=active, padding_value=0),
        ct.float32)
    logits = score + bias
    neg_inf = ct.full(shape=(BLOCK_N_C,), fill_value=float("-inf"), dtype=ct.float32)
    zero_f = ct.zeros((BLOCK_N_C,), dtype=ct.float32)
    logits_masked = ct.where(active, logits, neg_inf)

    row_max = ct.max(logits_masked)
    # If row is not live, output must be all zeros. Handle in the store.
    numer = ct.exp(logits_masked - row_max)
    numer = ct.where(active, numer, zero_f)
    denom = ct.sum(numer)
    # Avoid div-by-zero on rows where all are -inf (not live).
    one_f = ct.full(shape=(), fill_value=1.0, dtype=ct.float32)
    denom_safe = ct.where(denom > ct.full(shape=(), fill_value=0.0, dtype=ct.float32), denom, one_f)
    probs = ct.astype(numer / denom_safe, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_N_C,), dtype=ct.bfloat16)
    out = ct.where(active, probs, zero_bf)
    # For rows that are not live, output entire row of zeros.
    out = ct.where(row_live, out, zero_bf)

    # Store to 56-wide row: cols<56.
    col_store = cols < 56
    out_offs = row * 56 + cols
    ct.scatter(out_ptr, out_offs, out, mask=col_store)


@oracle_impl(hardware="B200", point="2ebbf10c")
@oracle_impl(hardware="B200", point="b57df06f")
@oracle_impl(hardware="B200", point="3551defb")
@oracle_impl(hardware="B200", point="7e00ce6b")
def oracle_forward(inputs):
    scores, rel_index, rel_table, *_shape_params = inputs
    heads = int(rel_table.shape[1])
    n_score_rows = int(scores.shape[0])
    n_rows = n_score_rows * 56
    device = scores.device

    out = torch.empty_strided(tuple(scores.shape), (3136, 56, 1),
                              device=device, dtype=torch.bfloat16)
    bias = torch.empty_strided((heads, 49, 49), (2401, 49, 1),
                               device=device, dtype=torch.bfloat16)

    scores_flat = scores.reshape(-1)
    rel_index_flat = rel_index.reshape(-1)
    rel_table_flat = rel_table.reshape(-1)
    bias_flat = bias.reshape(-1)
    out_flat = out.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(heads * 2401, BLOCK_BIAS), 1, 1),
        _materialize_relpos_bias_kernel,
        (rel_index_flat, rel_table_flat, bias_flat, heads, heads * 2401, BLOCK_BIAS),
    )
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _swin_softmax_kernel,
        (scores_flat, bias_flat, out_flat, n_rows, heads, BLOCK_N),
    )
    return out
