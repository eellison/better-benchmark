"""cuTile port of amax_sum_sum_c38064efac4d: GPT-J shifted-label cross-entropy.

Two kernels:
  1. Per-row online softmax over N_COLS=50400 -> amax, log_denom, loss, valid.
     Uses SHIFTED label (labels[row+1]) with ignore -100 semantics; loss falls
     back to arg2_1 fill value on invalid rows.
  2. Scalar mean-reduce over per-row loss & valid counts.
Also produces the padded label tensor (constant_pad_nd with 1 element).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_N = 8192


@ct.kernel
def _shifted_xent_stats_rows_kernel(
    logits_ptr,      # bf16 [ROWS, N_COLS] (contiguous)
    labels_ptr,      # i64 [SEQ_LEN]
    fill_ptr,        # f32 scalar
    padded_ptr,      # i64 [SEQ_LEN+1]
    amax_ptr,        # f32 [ROWS]
    log_ptr,         # f32 [ROWS]
    loss_ptr,        # f32 [ROWS]
    valid_ptr,       # f32 [ROWS]
    LOGITS_ROW_STRIDE: ct.Constant[int],
    N_COLS: ct.Constant[int],
    SEQ_LEN: ct.Constant[int],
    BLOCK_N_C: ct.Constant[int],
):
    row = ct.bid(0)
    row_i = ct.arange(1, dtype=ct.int64) + row

    # Copy labels_ptr[row] into padded_ptr[row]
    current_label = ct.gather(labels_ptr, row_i)
    ct.scatter(padded_ptr, row_i, current_label)

    # Load shifted label = labels_ptr[row+1] if row+1 < SEQ_LEN else -100
    has_next_val = row < (SEQ_LEN - 1)
    has_next = ct.full((1,), has_next_val, dtype=ct.bool_)
    next_off = ct.arange(1, dtype=ct.int64) + (row + 1)
    label_raw = ct.gather(labels_ptr, next_off, mask=has_next, padding_value=-100)
    minus100 = ct.full((1,), -100, dtype=ct.int64)
    is_valid = label_raw != minus100
    zero_i64 = ct.zeros((1,), dtype=ct.int64)
    safe_label = ct.where(is_valid, label_raw, zero_i64)

    # Target logit at [row, safe_label]
    target_off = safe_label + row * LOGITS_ROW_STRIDE
    target_bf = ct.gather(logits_ptr, target_off, mask=is_valid, padding_value=0.0)
    target = ct.astype(target_bf, ct.float32)

    # Online softmax over N_COLS
    row_max = ct.full((1,), -1.0e38, dtype=ct.float32)
    denom = ct.full((1,), 0.0, dtype=ct.float32)
    for block_start in range(0, N_COLS, BLOCK_N_C):
        cols = ct.arange(BLOCK_N_C, dtype=ct.int64) + block_start
        col_mask = cols < N_COLS
        zero_bn = ct.zeros((BLOCK_N_C,), dtype=ct.int64)
        safe_cols = ct.where(col_mask, cols, zero_bn)
        logits_off = row * LOGITS_ROW_STRIDE + safe_cols
        logits_bf = ct.gather(logits_ptr, logits_off)
        logits = ct.astype(logits_bf, ct.float32)
        neg_inf_bn = ct.full((BLOCK_N_C,), -1.0e38, dtype=ct.float32)
        logits_masked = ct.where(col_mask, logits, neg_inf_bn)
        block_max_scalar = ct.max(logits_masked)
        block_max_1 = ct.reshape(block_max_scalar, (1,))
        new_max = ct.where(row_max > block_max_1, row_max, block_max_1)
        denom = denom * ct.exp(row_max - new_max)
        new_max_broad = ct.broadcast_to(new_max, (BLOCK_N_C,))
        shifted = logits_masked - new_max_broad
        contrib = ct.exp(shifted)
        contrib = ct.where(col_mask, contrib,
                           ct.full((BLOCK_N_C,), 0.0, dtype=ct.float32))
        contrib_sum_scalar = ct.sum(contrib)
        contrib_sum_1 = ct.reshape(contrib_sum_scalar, (1,))
        denom = denom + contrib_sum_1
        row_max = new_max

    log_denom = ct.log(denom)
    # loss = row_max + log_denom - target
    loss = row_max + log_denom - target
    # Load fill scalar
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    ct.store(amax_ptr, index=(row,), tile=row_max)
    ct.store(log_ptr, index=(row,), tile=log_denom)
    zero_f_1 = ct.full((1,), 0.0, dtype=ct.float32)
    one_f_1 = ct.full((1,), 1.0, dtype=ct.float32)
    ct.store(loss_ptr, index=(row,), tile=ct.where(is_valid, loss, fill))
    ct.store(valid_ptr, index=(row,), tile=ct.where(is_valid, one_f_1, zero_f_1))


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    count_out_ptr,
    div_out_ptr,
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
    ct.store(count_out_ptr, index=(0,), tile=ct.reshape(total_valid, (1,)))
    ct.store(div_out_ptr, index=(0,),
             tile=ct.reshape(total_loss / total_valid, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="a71ab921")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)  # [1, 128, 50400]
    matrix_shape = _shape_tuple(shape_2d)  # [128, 50400] or [-1, 50400]
    n_cols = int(matrix_shape[1])
    n_rows = int(arg0_1.numel() // n_cols)
    seq_len = int(arg1_1.shape[1])
    device = arg0_1.device

    logits_view = arg0_1.view(view_shape)
    padded = torch.empty_strided(
        (int(arg1_1.shape[0]), seq_len + 1),
        (seq_len + 1, 1),
        device=device, dtype=torch.int64,
    )
    amax = torch.empty_strided((n_rows, 1), (1, 1), device=device, dtype=torch.float32)
    log = torch.empty_strided((n_rows, 1), (1, 1), device=device, dtype=torch.float32)
    loss_per_row = torch.empty(n_rows, device=device, dtype=torch.float32)
    valid_per_row = torch.empty(n_rows, device=device, dtype=torch.float32)
    count = torch.empty((), device=device, dtype=torch.float32)
    div = torch.empty((), device=device, dtype=torch.float32)

    labels_1d = arg1_1.view(-1).contiguous()
    logits_row_stride = int(arg0_1.stride(0))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _shifted_xent_stats_rows_kernel,
        (arg0_1.reshape(-1), labels_1d, arg2_1.view(1),
         padded.reshape(-1),
         amax.reshape(-1), log.reshape(-1),
         loss_per_row, valid_per_row,
         logits_row_stride, n_cols, seq_len, min(BLOCK_N, _next_power_of_2(n_cols))),
    )
    # Final pad element: constant_pad_nd adds one column with value -100 at position SEQ_LEN.
    padded[..., -1] = -100
    ct.launch(
        stream, (1, 1, 1), _mean_reduce_kernel,
        (loss_per_row, valid_per_row, count.view(1), div.view(1),
         n_rows, _next_power_of_2(n_rows)),
    )
    return logits_view, padded, amax, log, count, div
