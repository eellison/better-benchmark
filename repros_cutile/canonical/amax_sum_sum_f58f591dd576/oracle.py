"""cuTile port of amax_sum_sum_f58f591dd576: shifted causal-LM cross-entropy.

Fuses all the shifted-label prep + online logsumexp + target gather into one
kernel, matching Triton's structure. Removes the torch.full/copy/where prep
work that was polluting the launch path.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _shifted_xent_row_kernel(
    logits_ptr,      # bf16 (rows, N_COLS)
    labels_ptr,      # i64 (rows,) - flat labels
    padded_ptr,      # i64 (batch, PADDED_SEQ_LEN) - output padded labels
    amax_ptr,        # f32 (rows,)
    log_ptr,         # f32 (rows,)
    loss_ptr,        # f32 (rows,)
    valid_ptr,       # f32 (rows,)
    SEQ_LEN: ct.Constant[int],
    N_COLS: ct.Constant[int],
    N_BLOCKS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    batch = row // SEQ_LEN
    pos = row - batch * SEQ_LEN

    # Load current label; store to padded[batch, pos]. padded is (batch, SEQ_LEN+1).
    # The tile-space index for a (batch, SEQ_LEN+1) array with tile shape
    # (1, 1) is (batch, pos), matching the eager write pattern.
    cur_label_tile = ct.load(labels_ptr, index=(row,), shape=(1,))
    cur_label_2d = ct.reshape(cur_label_tile, (1, 1))
    ct.store(padded_ptr, index=(batch, pos), tile=cur_label_2d)
    # If pos == SEQ_LEN - 1, store -100 at padded[batch, SEQ_LEN].
    is_last_pos = pos == (SEQ_LEN - 1)
    if is_last_pos:
        neg100 = ct.full(shape=(1, 1), fill_value=-100, dtype=ct.int64)
        ct.store(padded_ptr, index=(batch, SEQ_LEN), tile=neg100)

    # Load the shifted label = labels[row+1] if pos < seq_len-1 else -100.
    is_shift_valid = pos < (SEQ_LEN - 1)
    # Load labels[row+1] safely — if we're at the last position, still read
    # something in bounds. row+1 <= n_rows-1 is only invalid when row == n_rows-1
    # which is only for the last batch's last pos. We can clamp to row for safety.
    safe_row = ct.where(is_shift_valid, row + 1, row)
    shifted_tile = ct.load(labels_ptr, index=(safe_row,), shape=(1,))
    shifted_label_raw = ct.reshape(shifted_tile, ())
    shifted_label = ct.where(is_shift_valid, shifted_label_raw, -100)
    is_valid = shifted_label != -100
    safe_label = ct.where(is_valid, shifted_label, 0)
    safe_label_i32 = ct.astype(safe_label, ct.int32)

    # Online logsumexp + target gather.
    row_max = ct.full(shape=(), fill_value=-3.4028234663852886e38, dtype=ct.float32)
    denom = ct.full(shape=(), fill_value=0.0, dtype=ct.float32)
    target = ct.full(shape=(), fill_value=0.0, dtype=ct.float32)

    for kb in range(N_BLOCKS):
        tile_bf = ct.load(
            logits_ptr, index=(row, kb), shape=(1, BLOCK_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile_f = ct.astype(tile_bf, ct.float32)
        ks = ct.arange(BLOCK_N, dtype=ct.int32) + kb * BLOCK_N
        in_bounds = ks < N_COLS

        block_max = ct.max(tile_f)
        new_max = ct.where(block_max > row_max, block_max, row_max)
        denom = denom * ct.exp(row_max - new_max)
        # Only sum in-bounds columns (OOB padded to -inf so exp is ~0).
        zero_tile = ct.zeros((1, BLOCK_N), dtype=ct.float32)
        exp_tile = ct.exp(tile_f - new_max)
        in_bounds_2d = ct.reshape(in_bounds, (1, BLOCK_N))
        exp_masked = ct.where(in_bounds_2d, exp_tile, zero_tile)
        denom = denom + ct.sum(exp_masked)
        row_max = new_max
        # Target gather.
        match = ks == safe_label_i32
        picked = ct.where(ct.reshape(match, (1, BLOCK_N)), tile_f, zero_tile)
        target = target + ct.sum(picked)

    log_denom = ct.log(denom)
    loss_raw = row_max + log_denom - target
    zero_scalar = ct.astype(0.0, ct.float32)
    loss_final = ct.where(is_valid, loss_raw, zero_scalar)
    valid_val = ct.where(is_valid,
                         ct.astype(1.0, ct.float32), zero_scalar)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(log_ptr, index=(row,), tile=ct.reshape(log_denom, (1,)))
    ct.store(loss_ptr, index=(row,), tile=ct.reshape(loss_final, (1,)))
    ct.store(valid_ptr, index=(row,), tile=ct.reshape(valid_val, (1,)))


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    count_out_ptr,
    div_out_ptr,
    BLOCK_M: ct.Constant[int],
):
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                     padding_mode=ct.PaddingMode.ZERO)
    valid = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,),
                    padding_mode=ct.PaddingMode.ZERO)
    total_loss = ct.sum(losses)
    total_valid = ct.sum(valid)
    mean = total_loss / total_valid
    ct.store(count_out_ptr, index=(0,), tile=ct.reshape(total_valid, (1,)))
    ct.store(div_out_ptr, index=(0,), tile=ct.reshape(mean, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="4137e900", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="ab0881e7", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="92fc17b6", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    n_cols = int(shape_2d[1])
    batch = int(labels.shape[0])
    seq_len = int(labels.shape[1])
    n_rows = batch * seq_len
    device = logits.device

    logits_view = logits.view(view_shape)
    logits_2d = logits_view.view(n_rows, n_cols)
    padded = torch.empty_strided(
        (batch, seq_len + 1), (seq_len + 1, 1),
        device=device, dtype=torch.int64,
    )
    amax = torch.empty_strided((n_rows, 1), (1, 1), device=device, dtype=torch.float32)
    log = torch.empty_strided((n_rows, 1), (1, 1), device=device, dtype=torch.float32)
    amax_1d = amax.view(n_rows)
    log_1d = log.view(n_rows)
    loss_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    count = torch.empty_strided((), (), device=device, dtype=torch.float32)
    div = torch.empty_strided((), (), device=device, dtype=torch.float32)

    labels_1d = labels.view(-1)

    block_n = min(BLOCK_N, 1 << (n_cols - 1).bit_length())
    n_blocks = (n_cols + block_n - 1) // block_n

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _shifted_xent_row_kernel,
        (logits_2d, labels_1d, padded, amax_1d, log_1d, loss_per_row, valid_per_row,
         seq_len, n_cols, n_blocks, block_n),
    )
    block_m = 1 << (n_rows - 1).bit_length()
    ct.launch(
        stream, (1, 1, 1), _mean_reduce_kernel,
        (loss_per_row, valid_per_row, count.view(1), div.view(1), block_m),
    )

    return logits_view, padded, amax, log, count, div
