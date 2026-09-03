"""cuTile port of amax_sum_ac4bab3e35d2: Swin relative-position softmax + pad.

Ports the Triton `_swin_shifted_relpos_softmax_pad_kernel` directly: one
program per (row) of the [N_ROWS, 56] flattened output; loads score, bias,
mask directly via strided views; softmax over the first 49 columns; writes
bf16 zero-padded output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_N = 64  # pow2 >= 49


@ct.kernel
def _swin_softmax_row_kernel(
    scores_ptr,     # bf16 [B_total, 56, 56]
    bias_ptr,       # bf16 [HEADS, 49, 49]
    mask_ptr,       # bf16 [MASK_WINDOWS, 49, 49]
    out_ptr,        # bf16 [B_total, 56, 56]
    HEADS: ct.Constant[int],
    MASK_WINDOWS: ct.Constant[int],
):
    row = ct.bid(0)      # 0..N_ROWS = B_total*56
    # Decompose row into (score_row, query) where score_row in [0, B_total),
    # query in [0, 56). Only query<49 rows are "live".
    score_row = row // 56
    query = row - score_row * 56
    window = score_row // HEADS
    head = score_row - window * HEADS
    mask_window = window - (window // MASK_WINDOWS) * MASK_WINDOWS

    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    col_active = cols < 49
    neg_inf = ct.full((BLOCK_N,), float("-inf"), dtype=ct.float32)
    zero_1d = ct.zeros((BLOCK_N,), dtype=ct.float32)

    # Load score row from [B_total, 56, 56] using tile shape (1,1,BLOCK_N)
    # at (score_row, query, 0) — cuTile handles OOB (cols>=56) via padding.
    score_bf = ct.load(
        scores_ptr, index=(score_row, query, 0), shape=(1, 1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    score = ct.astype(ct.reshape(score_bf, (BLOCK_N,)), ct.float32)

    bias_bf = ct.load(
        bias_ptr, index=(head, query, 0), shape=(1, 1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias_1d = ct.astype(ct.reshape(bias_bf, (BLOCK_N,)), ct.float32)

    mask_bf = ct.load(
        mask_ptr, index=(mask_window, query, 0), shape=(1, 1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mask_1d = ct.astype(ct.reshape(mask_bf, (BLOCK_N,)), ct.float32)

    row_live = query < 49
    logits = score + bias_1d + mask_1d
    logits = ct.where(col_active, logits, neg_inf)

    row_max = ct.max(logits)
    safe_max = ct.where(row_live, row_max, 0.0)
    numer = ct.exp(logits - safe_max)
    numer = ct.where(col_active, numer, zero_1d)
    denom = ct.sum(numer)
    safe_denom = ct.where(row_live, denom, 1.0)
    probs = numer / safe_denom
    probs = ct.where(col_active, probs, zero_1d)
    probs_bf = ct.astype(probs, ct.bfloat16)
    # Only store if row is live; else store zeros. But cuTile store writes
    # the whole tile — write zeros for non-live rows too.
    write = ct.where(row_live,
                     probs_bf,
                     ct.zeros((BLOCK_N,), dtype=ct.bfloat16))
    ct.store(out_ptr, index=(score_row, query, 0),
             tile=ct.reshape(write, (1, 1, BLOCK_N)))


def _launch(inputs):
    (
        scores,       # bf16 [B_total, 56, 56]
        rel_index,    # i64 [49, 49]
        rel_table,    # bf16 [169, HEADS]
        window_mask,  # bf16 [WINDOWS, 49, 49]
        *_shape_params,
    ) = inputs
    heads = int(rel_table.shape[1])
    windows = int(window_mask.shape[0])
    B_total = int(scores.shape[0])
    device = scores.device

    # Build the bias table: [HEADS, 49, 49] in torch (small — 2401 * HEADS elements).
    idx_flat = rel_index.view(-1)  # [2401]
    bias_2401_h = rel_table[idx_flat]        # bf16 [2401, HEADS]
    bias_49_49_h = bias_2401_h.view(49, 49, heads)
    bias_h_49_49 = bias_49_49_h.permute(2, 0, 1).contiguous()  # [HEADS, 49, 49]

    # Output: same shape/stride as scores, zero-initialized so trailing
    # rows/cols are zero.
    out = torch.zeros_like(scores)

    total_rows = B_total * 56
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_rows, 1, 1),
        _swin_softmax_row_kernel,
        (scores, bias_h_49_49, window_mask, out, heads, windows),
    )
    return out


@oracle_impl(hardware="B200", point="b78c8bdf")
@oracle_impl(hardware="B200", point="2839b6b9")
@oracle_impl(hardware="B200", point="3ef8438d")
def oracle_forward(inputs):
    return _launch(inputs)
