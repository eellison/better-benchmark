"""cuTile port of amax_sum_sum_5a80f2fc257a: Electra shifted-label cross-entropy.

Two cuTile kernels mirror the Triton oracle:
1. Per-row: online amax + log-sum-exp over N_COLS in-kernel, then per-row
   loss + valid_mask.
2. Scalar: reduce loss sum / valid_count into (count, div).
Target gather + padded_labels construction are hoisted to torch (as in
Triton's masked scalar load pattern).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
SEQ_LEN = 512
VOCAB_FULL = 30528
VOCAB = 30522
N_ROWS = BATCH * SEQ_LEN  # 32768


@ct.kernel
def _shifted_xent_rows_kernel(
    logits_ptr,      # bf16 [N_ROWS, LOGITS_STRIDE]
    targets_ptr,     # f32  [N_ROWS] pre-gathered target logits
    is_valid_ptr,    # f32  [N_ROWS] 1.0/0.0
    amax_ptr,        # f32  [N_ROWS]
    log_ptr,         # f32  [N_ROWS]
    loss_ptr,        # f32  [N_ROWS]
    valid_out_ptr,   # f32  [N_ROWS]
    N_COLS: ct.Constant[int],
    N_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    target = ct.astype(
        ct.load(targets_ptr, index=(row,), shape=(1,)),
        ct.float32,
    )
    target_scalar = ct.reshape(target, ())
    is_valid_v = ct.load(is_valid_ptr, index=(row,), shape=(1,))
    is_valid_scalar = ct.reshape(is_valid_v, ()) > 0.0

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
    loss_val = ct.astype(0.0, ct.float32) - logp
    zero_f = ct.astype(0.0, ct.float32)
    one_f = ct.astype(1.0, ct.float32)
    loss_masked = ct.where(is_valid_scalar, loss_val, zero_f)
    valid_val = ct.where(is_valid_scalar, one_f, zero_f)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(log_ptr, index=(row,), tile=ct.reshape(log_denom, (1,)))
    ct.store(loss_ptr, index=(row,), tile=ct.reshape(loss_masked, (1,)))
    ct.store(valid_out_ptr, index=(row,), tile=ct.reshape(valid_val, (1,)))


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,        # f32 [N_ROWS]
    valid_ptr,       # f32 [N_ROWS]
    count_out_ptr,   # f32 [1]
    div_out_ptr,     # f32 [1]
    N_ROWS_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    cols = ct.arange(BLOCK_M, dtype=ct.int32)
    mask = cols < N_ROWS_
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                     padding_mode=ct.PaddingMode.ZERO)
    valid = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,),
                    padding_mode=ct.PaddingMode.ZERO)
    zero_f = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    losses_m = ct.where(mask, losses, zero_f)
    valid_m = ct.where(mask, valid, zero_f)
    total_loss = ct.sum(losses_m)
    total_valid = ct.sum(valid_m)
    ct.store(count_out_ptr, index=(0,), tile=ct.reshape(total_valid, (1,)))
    ct.store(div_out_ptr, index=(0,), tile=ct.reshape(total_loss / total_valid, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(n):
    r = 1
    while r < n:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="a7052938", BLOCK_N=8192, row_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, row_warps: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    device = logits.device
    batch = int(labels.shape[0])
    seq_len = int(labels.shape[1])
    n_rows = batch * seq_len

    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(view_shape)
    logits_2d = logits_view.view(matrix_shape)

    # Build padded_labels = [labels, -100]
    padded_labels = torch.nn.functional.pad(labels, [0, 1], value=-100)
    shifted = padded_labels[:, 1:].contiguous().view(n_rows)
    is_valid = shifted != -100
    is_valid_f = is_valid.to(torch.float32)
    safe_labels = torch.where(is_valid, shifted, torch.zeros_like(shifted))
    # Pre-gather target logit
    targets = torch.gather(
        logits_2d.to(torch.float32), 1,
        safe_labels.unsqueeze(1).clamp(0, n_cols - 1)
    ).squeeze(1)

    amax = torch.empty_strided(
        (n_rows, 1), (1, 1), device=device, dtype=torch.float32,
    )
    log = torch.empty_strided(
        (n_rows, 1), (1, 1), device=device, dtype=torch.float32,
    )
    loss_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    count = torch.empty_strided((), (), device=device, dtype=torch.float32)
    div = torch.empty_strided((), (), device=device, dtype=torch.float32)

    block_cols = min(BLOCK_N, _next_pow2(n_cols))
    n_tiles = (n_cols + block_cols - 1) // block_cols

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _shifted_xent_rows_kernel,
        (logits_2d, targets, is_valid_f,
         amax.view(n_rows), log.view(n_rows),
         loss_per_row, valid_per_row,
         n_cols, n_tiles, block_cols),
    )
    block_m = _next_pow2(n_rows)
    ct.launch(
        stream, (1, 1, 1), _mean_reduce_kernel,
        (loss_per_row, valid_per_row, count.view(1), div.view(1),
         n_rows, block_m),
    )
    return logits_view, padded_labels, amax, log, count, div
