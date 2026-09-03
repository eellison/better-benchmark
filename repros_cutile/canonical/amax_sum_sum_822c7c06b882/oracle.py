"""cuTile port of amax_sum_sum_822c7c06b882: sliced-vocab ignore-index xent."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_N = 8192


def _view_shape(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _xent_rows_kernel(
    labels_ptr, logits_ptr, targets_ptr, loss_ptr, valid_ptr,
    N_COLS: ct.Constant[int],
    N_TILES: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    row = ct.bid(0)
    label = ct.load(labels_ptr, index=(row,), shape=(1,))
    label_scalar = ct.reshape(label, ())
    is_valid = label_scalar != -100

    target_tile = ct.load(targets_ptr, index=(row,), shape=(1,))
    target = ct.astype(target_tile, ct.float32)
    target_scalar = ct.reshape(target, ())

    # Online row-max and exp-sum
    row_max = ct.astype(-float("inf"), ct.float32)
    denom = ct.astype(0.0, ct.float32)
    for tile_i in ct.static_iter(range(N_TILES)):
        block_start = tile_i * BLOCK_COLS
        logits = ct.load(
            logits_ptr,
            index=(row, tile_i),
            shape=(1, BLOCK_COLS),
        )
        logits_f = ct.astype(logits, ct.float32)
        cols = ct.arange(BLOCK_COLS, dtype=ct.int32) + block_start
        col_mask = ct.reshape(cols < N_COLS, (1, BLOCK_COLS))
        neg_inf = ct.full((1, BLOCK_COLS), -float("inf"), dtype=ct.float32)
        logits_masked = ct.where(col_mask, logits_f, neg_inf)
        block_max = ct.max(logits_masked)
        new_max = ct.maximum(row_max, block_max)
        denom = denom * ct.exp(row_max - new_max)
        denom = denom + ct.sum(ct.exp(logits_masked - new_max))
        row_max = new_max

    logp = target_scalar - row_max - ct.log(denom)
    loss = ct.astype(0.0, ct.float32) - ct.astype(ct.astype(logp, ct.bfloat16), ct.float32)
    loss_masked = ct.where(is_valid, loss, ct.astype(0.0, ct.float32))
    valid_val = ct.where(is_valid, ct.astype(1.0, ct.float32), ct.astype(0.0, ct.float32))
    ct.store(loss_ptr, index=(row,), tile=ct.astype(ct.reshape(loss_masked, (1,)), ct.bfloat16))
    ct.store(valid_ptr, index=(row,), tile=ct.reshape(valid_val, (1,)))


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr, valid_ptr, out_ptr,
    N_ROWS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    cols = ct.arange(BLOCK_M, dtype=ct.int32)
    mask = cols < N_ROWS
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                     padding_mode=ct.PaddingMode.ZERO)
    valid = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,),
                    padding_mode=ct.PaddingMode.ZERO)
    losses = ct.astype(losses, ct.float32)
    valid = ct.astype(valid, ct.float32)

    zero_f = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    losses_m = ct.where(mask, losses, zero_f)
    valid_m = ct.where(mask, valid, zero_f)
    total_loss = ct.astype(ct.astype(ct.sum(losses_m), ct.bfloat16), ct.float32)
    total_valid = ct.astype(ct.astype(ct.sum(valid_m), ct.bfloat16), ct.float32)
    out_val = total_loss / total_valid
    ct.store(out_ptr, index=(0,), tile=ct.astype(ct.reshape(out_val, (1,)), ct.bfloat16))


@oracle_impl(hardware="B200", point="4445581e")
@oracle_impl(hardware="B200", point="f78f1ecd")
@oracle_impl(hardware="B200", point="e195ea0d")
def oracle_forward(inputs):
    labels, logits, shape_3d, shape_2d = inputs
    n_cols = int(shape_2d[1])
    logits_slice = logits[:, :n_cols].contiguous()
    logits_view = logits[:, :n_cols].view(_view_shape(shape_3d))
    logits_2d = logits_slice.view(_view_shape(shape_2d))
    labels_1d = labels.view(-1).to(torch.int64)
    n_rows = int(logits_2d.shape[0])
    device = logits.device

    loss_per_row = torch.empty((n_rows,), device=device, dtype=torch.bfloat16)
    valid_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    out_1d = torch.empty((1,), device=device, dtype=torch.bfloat16)
    out = out_1d.view(())

    block_cols = min(BLOCK_N, _next_pow2(n_cols))
    n_tiles = (n_cols + block_cols - 1) // block_cols
    padded_cols = n_tiles * block_cols
    if padded_cols > n_cols:
        padded_logits = torch.zeros((n_rows, padded_cols), device=device, dtype=logits_2d.dtype)
        padded_logits[:, :n_cols].copy_(logits_2d)
        used_logits = padded_logits
    else:
        used_logits = logits_2d

    # Pre-gather targets: safe_label per row (0 for ignored).
    safe_labels = torch.where(labels_1d != -100, labels_1d,
                              torch.zeros_like(labels_1d))
    targets = torch.gather(logits_2d, 1, safe_labels.unsqueeze(1).clamp(0, n_cols - 1)).squeeze(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _xent_rows_kernel,
        (labels_1d, used_logits, targets, loss_per_row, valid_per_row,
         n_cols, n_tiles, block_cols),
    )

    block_m = _next_pow2(n_rows)
    ct.launch(
        stream, (1, 1, 1), _mean_reduce_kernel,
        (loss_per_row, valid_per_row, out_1d, n_rows, block_m),
    )

    return logits_view, out
