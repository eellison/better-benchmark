"""cuTile port of amax_sum_sum_a1ef5e832cd1: DeBERTaV2 sliced-vocab XENT mean.

Mirrors the Triton kernel structure: kernel 1 does online row logsumexp,
masked-gather target, ignore-index masking, and bf16 loss; kernel 2 does
the bf16-rounded reduction and final mean division.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _view_shape(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    targets_ptr,
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
    target = ct.astype(target_tile, ct.float32)
    target_scalar = ct.reshape(target, ())

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

    logp = target_scalar - row_max - ct.log(denom)
    loss_f = ct.astype(0.0, ct.float32) - ct.astype(
        ct.astype(logp, ct.bfloat16), ct.float32
    )
    zero_f = ct.astype(0.0, ct.float32)
    loss_masked = ct.where(is_valid, loss_f, zero_f)
    one_f = ct.astype(1.0, ct.float32)
    valid_val = ct.where(is_valid, one_f, zero_f)
    ct.store(
        loss_ptr,
        index=(row,),
        tile=ct.astype(ct.reshape(loss_masked, (1,)), ct.bfloat16),
    )
    ct.store(valid_ptr, index=(row,), tile=ct.reshape(valid_val, (1,)))


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    out_ptr,
    N_ROWS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    cols = ct.arange(BLOCK_M, dtype=ct.int32)
    mask = cols < N_ROWS
    losses = ct.load(
        loss_ptr, index=(0,), shape=(BLOCK_M,), padding_mode=ct.PaddingMode.ZERO
    )
    valid = ct.load(
        valid_ptr, index=(0,), shape=(BLOCK_M,), padding_mode=ct.PaddingMode.ZERO
    )
    losses_f = ct.astype(losses, ct.float32)
    valid_f = ct.astype(valid, ct.float32)
    zero_f = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    losses_m = ct.where(mask, losses_f, zero_f)
    valid_m = ct.where(mask, valid_f, zero_f)
    total_loss = ct.astype(ct.astype(ct.sum(losses_m), ct.bfloat16), ct.float32)
    total_valid = ct.astype(ct.astype(ct.sum(valid_m), ct.bfloat16), ct.float32)
    out_val = total_loss / total_valid
    ct.store(
        out_ptr,
        index=(0,),
        tile=ct.astype(ct.reshape(out_val, (1,)), ct.bfloat16),
    )


# 9e036e55: DeBERTaV2 masked-LM bf16 logits [4096,128104] -> [:, :128100].
@oracle_impl(hardware="B200", point="9e036e55", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    labels, logits, shape_3d, shape_2d = inputs
    n_cols = int(shape_2d[1])
    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(_view_shape(shape_3d))
    logits_2d = logits_view.view(_view_shape(shape_2d))
    labels_1d = labels.view(-1)
    n_rows = int(logits_2d.shape[0])
    device = logits.device

    block_cols = min(BLOCK_N, _next_pow2(n_cols))
    n_tiles = (n_cols + block_cols - 1) // block_cols

    # Pre-gather target logits via torch (safe_label). This mirrors what
    # Triton does with a single masked scalar load. cuTile's per-row indexed
    # scalar load isn't supported inside a kernel with a data-dependent index,
    # so we hoist the gather to the launch site.
    safe_labels = torch.where(
        labels_1d != -100, labels_1d, torch.zeros_like(labels_1d)
    )
    targets = torch.gather(
        logits_2d, 1, safe_labels.unsqueeze(1).clamp(0, n_cols - 1)
    ).squeeze(1)

    loss_per_row = torch.empty((n_rows,), device=device, dtype=torch.bfloat16)
    valid_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    out_1d = torch.empty((1,), device=device, dtype=torch.bfloat16)
    out = out_1d.view(())

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _xent_rows_kernel,
        (labels_1d, logits_2d, targets, loss_per_row, valid_per_row,
         n_cols, n_tiles, block_cols),
    )

    block_m = _next_pow2(n_rows)
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, out_1d, n_rows, block_m),
    )

    return logits_view, out
