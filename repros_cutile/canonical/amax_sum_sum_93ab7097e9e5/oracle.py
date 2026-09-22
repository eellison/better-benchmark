"""cuTile port of amax_sum_sum_93ab7097e9e5: BERT-family sliced-vocab
ignore-index cross-entropy scalar.

Two-kernel port:
  1. Per-row online log-softmax + gather target log-prob at label. Emits
     per-row bf16 loss and validity (with the observable bf16 round-trip).
  2. Reduce all rows to bf16-rounded sum(loss) / sum(valid).

The 30522 vocab column count is not a power of 2, so we scan the row in
BLOCK_N-sized tiles with an online running-max/denom accumulator. Labels
of -100 are treated as ignore (zero loss, zero validity).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


IGNORE_INDEX = -100


@ct.kernel
def _xent_rows_kernel(
    labels_ptr,      # int64 [N_ROWS]
    logits_ptr,      # bf16 [N_ROWS * LOGITS_ROW_STRIDE] (flat)
    loss_ptr,        # f32 [N_ROWS]
    valid_ptr,       # f32 [N_ROWS]
    N_COLS: ct.Constant[int],
    LOGITS_ROW_STRIDE: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    label_idx_1 = ct.arange(1, dtype=ct.int64) + row
    label_tile = ct.gather(labels_ptr, label_idx_1)
    is_valid = label_tile != IGNORE_INDEX
    zero_i = ct.zeros((1,), dtype=ct.int64)
    safe_label = ct.where(is_valid, label_tile, zero_i)

    row_start_1 = ct.arange(1, dtype=ct.int64) + (row * LOGITS_ROW_STRIDE)
    target_idx = row_start_1 + safe_label
    target_bf = ct.gather(logits_ptr, target_idx)
    target = ct.astype(target_bf, ct.float32)

    # Online row-max + denom.
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
        block_max_scalar = ct.max(logits_masked)
        block_max_1 = ct.reshape(block_max_scalar, (1,))
        new_max = ct.where(row_max > block_max_1, row_max, block_max_1)
        denom = denom * ct.exp(row_max - new_max)
        new_max_bn = ct.broadcast_to(new_max, (BLOCK_N,))
        shifted = logits_masked - new_max_bn
        contrib = ct.exp(shifted)
        contrib = ct.where(col_mask, contrib,
                           ct.full((BLOCK_N,), 0.0, dtype=ct.float32))
        contrib_sum_1 = ct.reshape(ct.sum(contrib), (1,))
        denom = denom + contrib_sum_1
        row_max = new_max

    log_denom = ct.log(denom)
    logp = target - row_max - log_denom
    # Match Triton's bf16 round-trip on the negated log-prob boundary.
    neg_logp_bf = ct.astype(0.0 - logp, ct.bfloat16)
    loss = ct.astype(neg_logp_bf, ct.float32)

    zero_f_1 = ct.full((1,), 0.0, dtype=ct.float32)
    one_f_1 = ct.full((1,), 1.0, dtype=ct.float32)
    loss_masked = ct.where(is_valid, loss, zero_f_1)
    valid_masked = ct.where(is_valid, one_f_1, zero_f_1)

    row_idx_1 = ct.arange(1, dtype=ct.int64) + row
    ct.scatter(loss_ptr, row_idx_1, loss_masked)
    ct.scatter(valid_ptr, row_idx_1, valid_masked)


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,      # f32 [N_ROWS]
    valid_ptr,     # f32 [N_ROWS]
    out_ptr,       # bf16 []
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
    total_loss_f = ct.sum(losses)
    total_valid_f = ct.sum(valid)
    # Match Triton's bf16 round-trip on total loss and total valid.
    total_loss = ct.astype(ct.astype(total_loss_f, ct.bfloat16), ct.float32)
    total_valid = ct.astype(ct.astype(total_valid_f, ct.bfloat16), ct.float32)
    result = total_loss / total_valid
    result_bf = ct.astype(result, ct.bfloat16)
    ct.scatter(out_ptr, ct.arange(1, dtype=ct.int64),
               ct.reshape(result_bf, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="4b47cbc0", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="c9d2e15a", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(matrix_shape[0])
    device = logits.device

    # Slice logits[:, :n_cols] (dropping the padded 6 columns) — this stays as
    # a non-contiguous view we return as output 0 and read via a stride.
    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(view_shape)
    logits_flat = logits.reshape(-1)
    labels_flat = labels.reshape(-1)
    logits_row_stride = int(logits.stride(0))

    loss_per_row = torch.empty(n_rows, device=device, dtype=torch.float32)
    valid_per_row = torch.empty(n_rows, device=device, dtype=torch.float32)
    out = torch.empty((), device=device, dtype=torch.bfloat16)
    out_flat = out.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _xent_rows_kernel,
        (labels_flat, logits_flat, loss_per_row, valid_per_row,
         n_cols, logits_row_stride, BLOCK_N),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, out_flat, n_rows,
         _next_power_of_2(n_rows)),
    )
    return logits_view, out
