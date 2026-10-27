"""cuTile port of amax_sum_sum_42988b64e7f9 (NEW_PATTERN): DistillGPT2
shifted-label cross-entropy scalar.

Two kernels:
  1. Per-row online log-softmax + gather the log-prob at the label index.
     Emits per-row loss and validity.
  2. Reduce all rows to sum(loss) / sum(valid).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ_LEN = 512
VOCAB = 50257


@ct.kernel
def _shifted_xent_rows_kernel(
    logits_ptr,       # bf16 [ROWS * LOGITS_STRIDE]
    labels_ptr,       # int64 [BATCH * SEQ_LEN]
    loss_ptr,         # f32 [ROWS]
    valid_ptr,        # f32 [ROWS]
    LOGITS_ROW_STRIDE: ct.Constant[int],
    SEQ_LEN_C: ct.Constant[int],
    N_COLS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    pos = row - (row // SEQ_LEN_C) * SEQ_LEN_C
    has_next = pos < (SEQ_LEN_C - 1)

    # Load labels_ptr + row + 1 (single scalar)
    label_idx_1 = ct.arange(1, dtype=ct.int64) + (row + 1)
    label_tile = ct.gather(labels_ptr, label_idx_1)
    is_valid_tile = label_tile != -100
    zero_int_1 = ct.zeros((1,), dtype=ct.int64)
    safe_label = ct.where(is_valid_tile, label_tile, zero_int_1)

    row_start_1 = ct.arange(1, dtype=ct.int64) + (row * LOGITS_ROW_STRIDE)
    # Target: logits[row, safe_label]
    target_idx = row_start_1 + safe_label
    target_bf = ct.gather(logits_ptr, target_idx)
    target = ct.astype(target_bf, ct.float32)

    # Online softmax over N_COLS in chunks of BLOCK_N.
    row_max = ct.full((1,), -1.0e38, dtype=ct.float32)
    denom = ct.full((1,), 0.0, dtype=ct.float32)
    for block_start in range(0, N_COLS, BLOCK_N):
        cols = ct.arange(BLOCK_N, dtype=ct.int64) + block_start
        col_mask = cols < N_COLS
        zero_bn = ct.zeros((BLOCK_N,), dtype=ct.int64)
        safe_cols = ct.where(col_mask, cols, zero_bn)
        logits_off = row * LOGITS_ROW_STRIDE + safe_cols
        logits_bf = ct.gather(logits_ptr, logits_off)
        logits = ct.astype(logits_bf, ct.float32)
        neg_inf_bn = ct.full((BLOCK_N,), -1.0e38, dtype=ct.float32)
        logits_masked = ct.where(col_mask, logits, neg_inf_bn)
        block_max_scalar = ct.max(logits_masked)  # returns tile shape ()?
        # We need 1-d [1] for combining with row_max.
        block_max_1 = ct.reshape(block_max_scalar, (1,))
        new_max = ct.where(row_max > block_max_1, row_max, block_max_1)
        # denom = denom * exp(row_max - new_max) + sum(exp(logits - new_max))
        denom = denom * ct.exp(row_max - new_max)
        new_max_broad = ct.broadcast_to(new_max, (BLOCK_N,))
        shifted = logits_masked - new_max_broad
        contrib = ct.exp(shifted)
        contrib = ct.where(col_mask, contrib,
                            ct.full((BLOCK_N,), 0.0, dtype=ct.float32))
        contrib_sum_scalar = ct.sum(contrib)
        contrib_sum_1 = ct.reshape(contrib_sum_scalar, (1,))
        denom = denom + contrib_sum_1
        row_max = new_max

    log_denom = ct.log(denom)
    logp = target - row_max - log_denom
    loss = -logp
    # If not valid, loss = 0
    zero_f_1 = ct.full((1,), 0.0, dtype=ct.float32)
    loss_masked = ct.where(is_valid_tile, loss, zero_f_1)
    one_f_1 = ct.full((1,), 1.0, dtype=ct.float32)
    valid_masked = ct.where(is_valid_tile, one_f_1, zero_f_1)

    # Store scalar loss and valid.
    row_idx_1 = ct.arange(1, dtype=ct.int64) + row
    ct.scatter(loss_ptr, row_idx_1, loss_masked)
    ct.scatter(valid_ptr, row_idx_1, valid_masked)


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    out_ptr,
    N_ROWS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    offsets = ct.arange(BLOCK_M, dtype=ct.int64)
    row_mask = offsets < N_ROWS
    zero_bm = ct.zeros((BLOCK_M,), dtype=ct.int64)
    safe = ct.where(row_mask, offsets, zero_bm)
    losses = ct.gather(loss_ptr, safe)
    valid = ct.gather(valid_ptr, safe)
    zero_bmf = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    losses = ct.where(row_mask, losses, zero_bmf)
    valid = ct.where(row_mask, valid, zero_bmf)
    total_loss = ct.sum(losses)
    total_valid = ct.sum(valid)
    result = total_loss / total_valid
    # Store scalar at index 0.
    ct.scatter(out_ptr, ct.arange(1, dtype=ct.int64),
                ct.reshape(result, (1,)))


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="c1ee6cd0", BLOCK_N=4096)
def oracle_forward(inputs, *, BLOCK_N: int):
    labels, logits, shape_2d = inputs
    matrix_shape = tuple(int(dim) for dim in shape_2d)
    n_cols = int(matrix_shape[1])
    seq_len = int(labels.shape[1])
    n_rows = int(labels.numel())

    logits_flat = logits.reshape(-1)
    labels_flat = labels.reshape(-1)

    loss_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    out_flat = out.view(1)

    stream = torch.cuda.current_stream()
    # logits[view] stride0 is n_cols (row-major); reshape returns contiguous.
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _shifted_xent_rows_kernel,
        (logits_flat, labels_flat, loss_per_row, valid_per_row,
         n_cols, seq_len, n_cols, BLOCK_N),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, out_flat, n_rows, _next_power_of_2(n_rows)),
    )
    return out
