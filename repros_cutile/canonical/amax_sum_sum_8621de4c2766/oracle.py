"""cuTile port of amax_sum_sum_8621de4c2766: T5/MT5 ignore-index cross-entropy.

Mirrors the Triton kernel structure: kernel 1 does online row logsumexp,
masked-gather target, ignore-index masking (with a fallback fill scalar), and
per-row loss + valid indicator; kernel 2 does the sum-and-mean reduce.
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
def _xent_stats_rows_kernel(
    logits_ptr,
    targets_ptr,     # f32 [n_rows] pre-gathered
    labels_ptr,
    fallback_ptr,
    amax_ptr,
    log_ptr,
    loss_ptr,
    valid_ptr,
    N_COLS: ct.Constant[int],
    N_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    label = ct.load(labels_ptr, index=(row,), shape=(1,))
    label_scalar = ct.reshape(label, ())
    is_valid = label_scalar != -100

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
    # Match Triton: bf16 rounding on logp then back to f32, then negate.
    logp_bf = ct.astype(logp, ct.bfloat16)
    logp_bf_f = ct.astype(logp_bf, ct.float32)
    loss = ct.astype(0.0, ct.float32) - logp_bf_f

    fallback_tile = ct.load(fallback_ptr, index=(0,), shape=(1,))
    fallback_scalar = ct.astype(ct.reshape(fallback_tile, ()), ct.float32)

    final_loss = ct.where(is_valid, loss, fallback_scalar)
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
    ct.store(
        div_out_ptr, index=(0,), tile=ct.reshape(total_loss / total_valid, (1,))
    )


@oracle_impl(hardware="B200", point="0b78f8f6", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="5d25ba41", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, labels, fallback, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    n_rows = int(labels.numel())
    n_cols = int(shape_2d[1])
    device = logits.device

    logits_view = logits.view(view_shape)
    logits_2d = logits_view.view(n_rows, n_cols)
    labels_1d = labels.view(-1)

    # Pre-gather target logits with safe label.
    safe_labels = torch.where(
        labels_1d != -100, labels_1d, torch.zeros_like(labels_1d)
    )
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
        _xent_stats_rows_kernel,
        (
            logits_2d,
            targets,
            labels_1d,
            fallback.view(1),
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
        (
            loss_per_row,
            valid_per_row,
            count.view(1),
            div.view(1),
            n_rows,
            _next_pow2(n_rows),
        ),
    )

    return logits_view, amax, log, count, div
