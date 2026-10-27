"""cuTile port of amax_sum_sum_a4f202093213: GPTNeo shifted-label cross-entropy.

Mirrors the Triton kernel structure: kernel 1 does the online row logsumexp,
masked target gather, and per-row loss (with fill fallback for ignored rows);
kernel 2 does the loss sum plus mean division. Padded labels are built via
a metadata-only torch pad (Triton stores the same tensor from inside kernel 1;
we hoist it to a single torch op to avoid extra kernel work).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _shifted_xent_stats_kernel(
    logits_ptr,
    targets_ptr,     # f32 [n_rows] pre-gathered target values
    valid_mask_ptr,  # b8  [n_rows] shifted label != -100
    fill_ptr,
    amax_ptr,
    log_ptr,
    loss_ptr,
    valid_ptr,
    N_COLS: ct.Constant[int],
    N_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    is_valid_tile = ct.load(valid_mask_ptr, index=(row,), shape=(1,))
    is_valid = ct.reshape(is_valid_tile, ())

    target_tile = ct.load(targets_ptr, index=(row,), shape=(1,))
    target_scalar = ct.reshape(target_tile, ())

    row_max = ct.astype(-float("inf"), ct.float32)
    denom = ct.astype(0.0, ct.float32)
    for tile_i in ct.static_iter(range(N_TILES)):
        block_start = tile_i * BLOCK_N
        logits = ct.load(
            logits_ptr,
            index=(row, tile_i),
            shape=(1, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        logits_f = ct.astype(logits, ct.float32)
        cols = ct.arange(BLOCK_N, dtype=ct.int32) + block_start
        col_mask = ct.reshape(cols < N_COLS, (1, BLOCK_N))
        neg_inf = ct.full((1, BLOCK_N), -float("inf"), dtype=ct.float32)
        logits_masked = ct.where(col_mask, logits_f, neg_inf)
        block_max = ct.max(logits_masked)
        new_max = ct.maximum(row_max, block_max)
        denom = denom * ct.exp(row_max - new_max)
        denom = denom + ct.sum(ct.exp(logits_masked - new_max))
        row_max = new_max

    log_denom = ct.log(denom)
    logp = target_scalar - row_max - log_denom
    loss = ct.astype(0.0, ct.float32) - logp

    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar = ct.reshape(fill_tile, ())

    final_loss = ct.where(is_valid, loss, fill_scalar)
    valid_val = ct.where(
        is_valid, ct.astype(1.0, ct.float32), ct.astype(0.0, ct.float32)
    )

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(log_ptr, index=(row,), tile=ct.reshape(log_denom, (1,)))
    ct.store(loss_ptr, index=(row,), tile=ct.reshape(final_loss, (1,)))
    ct.store(valid_ptr, index=(row,), tile=ct.reshape(valid_val, (1,)))


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    count_out_ptr,
    div_out_ptr,
    N_ROWS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    losses = ct.load(
        loss_ptr, index=(0,), shape=(BLOCK_M,), padding_mode=ct.PaddingMode.ZERO
    )
    valid = ct.load(
        valid_ptr, index=(0,), shape=(BLOCK_M,), padding_mode=ct.PaddingMode.ZERO
    )
    losses_f = ct.astype(losses, ct.float32)
    valid_f = ct.astype(valid, ct.float32)
    cols = ct.arange(BLOCK_M, dtype=ct.int32)
    mask = cols < N_ROWS
    zero_f = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    losses_m = ct.where(mask, losses_f, zero_f)
    valid_m = ct.where(mask, valid_f, zero_f)
    total_loss = ct.sum(losses_m)
    total_valid = ct.sum(valid_m)
    ct.store(count_out_ptr, index=(0,), tile=ct.reshape(total_valid, (1,)))
    ct.store(div_out_ptr, index=(0,), tile=ct.reshape(total_loss / total_valid, (1,)))


# 64c4bb0f: GPTNeo train shifted labels i64[32,128], bf16 logits [4096,50264].
@oracle_impl(hardware="B200", point="64c4bb0f", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, labels, fill, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    batch = int(labels.shape[0])
    seq_len = int(labels.shape[1])
    n_rows = batch * seq_len
    device = logits.device

    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(view_shape)
    logits_2d = logits_view.view(matrix_shape)

    # Metadata-only: pad labels with -100 on the right, mirroring
    # constant_pad_nd. This is the padded_labels return value.
    padded_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    # Shifted labels: labels[b, pos+1] with -100 boundary at pos=SEQ_LEN-1.
    shifted_labels = padded_labels[:, 1:].contiguous().view(-1)
    valid_mask = shifted_labels != -100
    safe_labels = torch.where(valid_mask, shifted_labels, torch.zeros_like(shifted_labels))
    targets = torch.gather(
        logits_2d.to(torch.float32), 1, safe_labels.unsqueeze(1).clamp(0, n_cols - 1)
    ).squeeze(1)

    amax = torch.empty_strided(
        (n_rows, 1), (1, 1), device=device, dtype=torch.float32
    )
    log = torch.empty_strided(
        (n_rows, 1), (1, 1), device=device, dtype=torch.float32
    )
    loss_per_row = torch.empty_strided(
        (n_rows,), (1,), device=device, dtype=torch.float32
    )
    valid_per_row = torch.empty_strided(
        (n_rows,), (1,), device=device, dtype=torch.float32
    )
    count = torch.empty_strided((), (), device=device, dtype=torch.float32)
    div = torch.empty_strided((), (), device=device, dtype=torch.float32)

    block_cols = min(BLOCK_N, _next_pow2(n_cols))
    n_tiles = (n_cols + block_cols - 1) // block_cols

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _shifted_xent_stats_kernel,
        (
            logits_2d,
            targets,
            valid_mask,
            fill.view(1),
            amax.view(-1),
            log.view(-1),
            loss_per_row,
            valid_per_row,
            n_cols,
            n_tiles,
            block_cols,
        ),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, count.view(1), div.view(1),
         n_rows, _next_pow2(n_rows)),
    )
    return logits_view, padded_labels, amax, log, count, div
