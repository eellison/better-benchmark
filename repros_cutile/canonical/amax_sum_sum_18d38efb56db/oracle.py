"""cuTile port of amax_sum_sum_18d38efb56db: DeBERTaV2 sliced-vocab
ignore-index cross-entropy scalar.

Two kernels:
  1. Per-row online softmax over 128100 cols -> amax, log_denom, per-row loss,
     per-row valid indicator. Loss goes through bf16 rounding before the gather.
  2. Reduce to scalar count and mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xent_stats_rows_kernel(
    logits_ptr,      # bf16 [ROWS, LOGITS_STRIDE]  (from 4096 x 128104 tensor)
    labels_ptr,      # i64  [ROWS]
    amax_ptr,        # f32  [ROWS]
    log_ptr,         # f32  [ROWS]
    loss_ptr,        # f32  [ROWS]
    valid_ptr,       # f32  [ROWS]
    LOGITS_ROW_STRIDE: ct.Constant[int],
    N_COLS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    label_idx_1 = ct.arange(1, dtype=ct.int64) + row
    label = ct.gather(labels_ptr, label_idx_1)
    minus100 = ct.full((1,), -100, dtype=ct.int64)
    is_valid = label != minus100
    zero_i64 = ct.zeros((1,), dtype=ct.int64)
    safe_label = ct.where(is_valid, label, zero_i64)

    # Target logit at [row, safe_label]
    target_off = safe_label + row * LOGITS_ROW_STRIDE
    target_bf = ct.gather(logits_ptr, target_off)
    target = ct.astype(target_bf, ct.float32)

    # Online softmax over N_COLS
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
    # logp = target - row_max - log_denom, then rounded to bf16 then back to f32
    logp = target - row_max - log_denom
    logp_bf16 = ct.astype(logp, ct.bfloat16)
    logp_f32 = ct.astype(logp_bf16, ct.float32)
    loss = -logp_f32

    zero_f_1 = ct.full((1,), 0.0, dtype=ct.float32)
    one_f_1 = ct.full((1,), 1.0, dtype=ct.float32)
    loss_masked = ct.where(is_valid, loss, zero_f_1)
    valid_val = ct.where(is_valid, one_f_1, zero_f_1)

    ct.store(amax_ptr, index=(row,), tile=row_max)
    ct.store(log_ptr, index=(row,), tile=log_denom)
    ct.store(loss_ptr, index=(row,), tile=loss_masked)
    ct.store(valid_ptr, index=(row,), tile=valid_val)


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


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 9e036e55: DeBERTaV2 masked-LM train, bf16 logits [4096,128104] -> [:, :128100].
@oracle_impl(hardware="B200", point="9e036e55", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)  # [8, 512, 128100]
    matrix_shape = _shape_tuple(shape_2d)  # [4096, 128100]
    n_rows = int(matrix_shape[0])
    n_cols = int(matrix_shape[1])

    # The strided slice/view: view_shape uses logits[:, :n_cols] with stride 128104.
    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(view_shape)
    labels_1d = labels.view(-1).contiguous()

    amax = torch.empty_strided(
        (n_rows, 1), (1, 1), device=logits.device, dtype=torch.float32,
    )
    log = torch.empty_strided(
        (n_rows, 1), (1, 1), device=logits.device, dtype=torch.float32,
    )
    loss_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    count = torch.empty((), device=logits.device, dtype=torch.float32)
    div = torch.empty((), device=logits.device, dtype=torch.float32)

    # Row stride of the (raw) logits tensor.
    logits_row_stride = int(logits.stride(0))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _xent_stats_rows_kernel,
        (logits.reshape(-1), labels_1d,
         amax.reshape(-1), log.reshape(-1),
         loss_per_row, valid_per_row,
         logits_row_stride, n_cols, BLOCK_N),
    )
    ct.launch(
        stream, (1, 1, 1), _mean_reduce_kernel,
        (loss_per_row, valid_per_row, count.view(1), div.view(1),
         n_rows, _next_power_of_2(n_rows)),
    )
    return logits_view, amax, log, count, div
