"""cuTile port of amax_sum_sum_c94a0970300f: shifted-label causal-LM CE mean.

Emulates Repro.forward:
  - Compute row-wise log-softmax over bf16-cast-to-f32 logits (dynamic vocab).
  - Gather target logit per row using shifted labels (with -100 ignore index).
  - Return the reshaped logits view alias and the scalar mean loss.

The vocab dimension is not power-of-2, so we round up to the next power of 2
and mask via a col_mask, filling with -inf for max and 0 for sum reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _shifted_xent_row_kernel(
    labels_ptr,     # i64 [n_rows]  (shifted labels, -100 for ignored)
    logits_ptr,     # bf16 [n_rows, N_COLS]
    loss_ptr,       # f32 [n_rows]
    valid_ptr,      # f32 [n_rows]
    N_COLS: ct.Constant[int],
    N_PAD: ct.Constant[int],
):
    row = ct.bid(0)

    label = ct.load(labels_ptr, index=(row,), shape=(1,))
    is_valid = label != -100
    zero_i = ct.full(shape=(1,), fill_value=0, dtype=ct.int64)
    safe_label = ct.where(is_valid, label, zero_i)

    logits_bf = ct.load(
        logits_ptr,
        index=(row, 0),
        shape=(1, N_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits = ct.astype(logits_bf, ct.float32)
    cols = ct.arange(N_PAD, dtype=ct.int64)
    col_mask = cols < N_COLS
    col_mask_2d = ct.reshape(col_mask, (1, N_PAD))

    # Masked max: OOB → -inf.
    neg_inf = ct.full((1, N_PAD), -1.0e38, dtype=ct.float32)
    masked_for_max = ct.where(col_mask_2d, logits, neg_inf)
    row_max = ct.max(masked_for_max, axis=1, keepdims=True)
    shifted = logits - row_max
    exp_vals = ct.exp(shifted)
    # Zero OOB before sum.
    zeros = ct.full((1, N_PAD), 0.0, dtype=ct.float32)
    exp_masked = ct.where(col_mask_2d, exp_vals, zeros)
    denom = ct.sum(exp_masked, axis=1, keepdims=True)
    log_denom = ct.log(denom)

    # Extract target logit via one-hot mul.
    safe_label_2d = ct.reshape(safe_label, (1, 1))
    match = cols == ct.reshape(safe_label_2d, (1, N_PAD)) if False else None  # dead
    # cleanly: build (1, N_PAD) one-hot
    cols_2d = ct.reshape(cols, (1, N_PAD))
    one_hot = cols_2d == safe_label_2d
    one_hot_f = ct.astype(one_hot, ct.float32)
    target_logit = ct.sum(one_hot_f * logits, axis=1, keepdims=True)

    logp = (target_logit - row_max) - log_denom
    loss = -logp

    is_valid_2d = ct.reshape(is_valid, (1, 1))
    loss_masked = ct.where(is_valid_2d, loss, zeros)  # zeros is (1, N_PAD) - just use scalar 0
    zero_1x1 = ct.full((1, 1), 0.0, dtype=ct.float32)
    loss_out = ct.where(is_valid_2d, loss, zero_1x1)
    valid_f = ct.astype(is_valid_2d, ct.float32)

    ct.store(loss_ptr, index=(row,), tile=ct.reshape(loss_out, (1,)))
    ct.store(valid_ptr, index=(row,), tile=ct.reshape(valid_f, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="4137e900", BLOCK_N=4096)
@oracle_impl(hardware="B200", point="92fc17b6", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="ab0881e7", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(logits.numel() // n_cols)
    seq_len = int(labels.shape[1])
    device = logits.device

    logits_view = logits.view(view_shape)
    logits_2d = logits_view.view(n_rows, n_cols)

    # Build shifted labels [n_rows]: pad labels by -100 to the right and slice.
    padded_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shifted_labels = padded_labels[:, 1:].contiguous().view(-1)

    loss_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)

    n_pad = _next_pow2(n_cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _shifted_xent_row_kernel,
        (shifted_labels, logits_2d, loss_per_row, valid_per_row, n_cols, n_pad),
    )

    out = torch.empty_strided((), (), device=device, dtype=torch.float32)
    out.fill_(0.0)
    total_loss = loss_per_row.sum(dtype=torch.float32)
    total_valid = valid_per_row.sum(dtype=torch.float32)
    out.copy_(total_loss / total_valid)

    return logits_view, out
