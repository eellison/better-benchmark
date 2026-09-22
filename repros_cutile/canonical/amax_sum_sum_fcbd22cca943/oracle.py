"""cuTile port of amax_sum_sum_fcbd22cca943: GPT-J shifted-label
cross-entropy mean over a bf16 logits tensor with 50400 vocabulary.

Row kernel: online logsumexp over BLOCK_N tiles; second kernel reduces to
scalar mean. The dynamic label gather is done in torch before dispatch to
keep the kernel focused on the reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_logsumexp_kernel(
    logits_ptr,     # bf16 (rows, N_COLS)
    target_ptr,     # f32 (rows,) — pre-gathered target logit per row
    valid_mask_ptr, # f32 (rows,) — 1.0 if row is valid, else 0.0
    loss_ptr,       # f32 (rows,)
    valid_ptr,      # f32 (rows,)
    N_COLS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    n_blocks = (N_COLS + BLOCK_N - 1) // BLOCK_N
    running_max = ct.full((1,), fill_value=-1e38, dtype=ct.float32)
    running_denom = ct.full((1,), fill_value=0.0, dtype=ct.float32)
    for b in range(n_blocks):
        logits = ct.load(
            logits_ptr, index=(row, b), shape=(1, BLOCK_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        logits_f = ct.astype(logits, ct.float32)
        block_max_scalar = ct.max(logits_f)
        block_max_1 = ct.reshape(block_max_scalar, (1,))
        new_max = ct.maximum(running_max, block_max_1)
        exp_shift = ct.exp(running_max - new_max)
        new_max_2d = ct.reshape(new_max, (1, 1))
        exp_logits = ct.exp(logits_f - new_max_2d)
        block_sum = ct.sum(exp_logits)
        running_denom = running_denom * exp_shift + ct.reshape(block_sum, (1,))
        running_max = new_max

    target = ct.load(target_ptr, index=(row,), shape=(1,))
    valid_mask = ct.load(valid_mask_ptr, index=(row,), shape=(1,))
    loss = running_max + ct.log(running_denom) - target
    # If invalid, force loss=0 and valid=0.
    loss_final = loss * valid_mask
    ct.store(loss_ptr, index=(row,), tile=loss_final)
    ct.store(valid_ptr, index=(row,), tile=valid_mask)


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    out_ptr,
    N_ROWS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                     padding_mode=ct.PaddingMode.ZERO)
    valid = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,),
                    padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(losses) / ct.sum(valid)
    ct.store(out_ptr, index=(0,), tile=ct.reshape(total, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="627fbc0f", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(logits.numel() // n_cols)
    seq_len = int(labels.shape[1])
    device = logits.device

    logits_view = logits.view(view_shape)

    # Pre-gather target and validity in torch.
    # For row `r` in [0, n_rows): position = r % seq_len; if position < seq_len-1,
    # target label is labels_flat[r+1]; else -100 (invalid).
    labels_flat = labels.view(-1)  # (seq_len,)
    # Build shifted labels: labels_flat[1:] padded with -100.
    # But we don't know n_rows relative to seq_len — assume n_rows is a multiple.
    # Actually n_rows = numel(logits) / n_cols. If matches (batch*seq), okay.
    # For simplicity: n_rows == seq_len here (batch=1).
    labels_shift = torch.full((n_rows,), -100, dtype=torch.int64, device=device)
    # For each position pos < seq_len-1: labels_shift[pos] = labels_flat[pos+1]
    if n_rows == seq_len:
        labels_shift[: seq_len - 1] = labels_flat[1:]
    else:
        # General case (interleaved seqs)
        for b in range(n_rows // seq_len):
            base = b * seq_len
            labels_shift[base:base + seq_len - 1] = labels_flat[base + 1:base + seq_len]
    valid_mask = (labels_shift != -100).to(torch.float32)
    safe_label = torch.where(labels_shift >= 0, labels_shift, torch.zeros_like(labels_shift))
    # Gather target logits: logits[row, safe_label[row]]
    row_idx = torch.arange(n_rows, device=device)
    target = logits[row_idx, safe_label].to(torch.float32)

    loss_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    out = torch.empty((), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _row_logsumexp_kernel,
        (logits, target, valid_mask, loss_per_row, valid_per_row, n_cols, BLOCK_N),
    )
    block_m = 1 << (n_rows - 1).bit_length()
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, out.view(1), n_rows, block_m),
    )
    return logits_view, out
