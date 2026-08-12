"""cuTile port of amax_sum_sum_6aa38ad4af60: ignore-index cross-entropy mean.

Computes per-row log-softmax + gathered target loss + ignore mask, then the
final scalar mean = sum(loss) / sum(valid).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xent_rows_kernel(
    labels_ptr,        # i64 [ROWS]
    logits_ptr,        # f32 [ROWS, N_COLS] (padded)
    target_col_ptr,    # f32 [ROWS] (gathered logit at safe label col)
    loss_ptr,          # f32 [ROWS]
    valid_ptr,         # f32 [ROWS]
    N_COLS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    label_tile = ct.load(labels_ptr, index=(row,), shape=(1,))
    is_valid_tile = label_tile != ct.full(shape=(1,), fill_value=-100, dtype=ct.int64)

    # Load the row of logits (padded to BLOCK_N with -inf via padding mask)
    logits = ct.load(logits_ptr, index=(row, 0), shape=(1, BLOCK_N))
    logits_f = ct.astype(logits, ct.float32)

    # Row max/sum for logsumexp; mask off padded columns
    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask = ct.reshape(cols < N_COLS, (1, BLOCK_N))
    logits_masked = ct.where(col_mask, logits_f, -float("inf"))
    row_max = ct.max(logits_masked)
    denom = ct.sum(ct.where(col_mask, ct.exp(logits_f - row_max), 0.0))
    log_denom = ct.log(denom)

    target = ct.load(target_col_ptr, index=(row,), shape=(1,))
    target_f = ct.astype(target, ct.float32)
    log_prob = (target_f - row_max) - log_denom
    loss = -log_prob

    ct.store(
        loss_ptr,
        index=(row,),
        tile=ct.where(is_valid_tile, loss, ct.full(shape=(1,), fill_value=0.0, dtype=ct.float32)),
    )
    ct.store(
        valid_ptr,
        index=(row,),
        tile=ct.where(is_valid_tile, ct.full(shape=(1,), fill_value=1.0, dtype=ct.float32), ct.full(shape=(1,), fill_value=0.0, dtype=ct.float32)),
    )


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr, valid_ptr, out_ptr, N_ROWS: ct.Constant[int], BLOCK_M: ct.Constant[int]
):
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,))
    valid = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,))
    rows = ct.arange(BLOCK_M, dtype=ct.int32)
    row_mask = rows < N_ROWS
    total_loss = ct.sum(ct.where(row_mask, losses, 0.0))
    total_valid = ct.sum(ct.where(row_mask, valid, 0.0))
    ct.store(out_ptr, index=(0,), tile=(total_loss / total_valid))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@oracle_impl(hardware="B200", point="a49a430b", BLOCK_N=32768)
def oracle_forward(inputs, *, BLOCK_N: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(logits.numel() // n_cols)

    labels_1d = labels.view(-1).contiguous()

    # Pre-gather target logits (avoids indexing inside kernel)
    safe_label = torch.where(labels_1d != -100, labels_1d, torch.zeros_like(labels_1d))
    row_idx = torch.arange(n_rows, device=logits.device, dtype=torch.int64)
    target_gathered = logits[row_idx, safe_label].contiguous()

    logits_view = logits.view(view_shape)

    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((1,), device=logits.device, dtype=torch.float32)

    # Pad logits row-wise to BLOCK_N power-of-two width
    if BLOCK_N < n_cols:
        BLOCK_N = _next_pow2(n_cols)
    if BLOCK_N > n_cols:
        pad = torch.full(
            (n_rows, BLOCK_N - n_cols),
            -float("inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        logits_padded = torch.cat([logits, pad], dim=1).contiguous()
    else:
        logits_padded = logits.contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _xent_rows_kernel,
        (labels_1d, logits_padded, target_gathered, loss_per_row, valid_per_row, n_cols, BLOCK_N),
    )
    BLOCK_M = _next_pow2(n_rows)
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, out, n_rows, BLOCK_M),
    )
    return logits_view, out.view(())
