"""cuTile port of amax_sum_sum_a97be2219d64: PLBart sliced-vocabulary
cross-entropy.

Row cross-entropy over logits[16384, 50005] against labels[16384] with
ignore_index=-100. Fused online logsumexp iterates the K dim in
BLOCK_N=8192 chunks — no need to copy the sliced view, since the K
dim of the logits array is the padded 50008; we mask OOB in the last
block using arange >= n_cols.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xent_row_kernel(
    labels_ptr,   # i64 [N]
    logits_ptr,   # bf16 [N, K_PADDED]  (raw logits, not sliced)
    loss_ptr,     # bf16 [N]
    valid_ptr,    # f32 [N]
    N_COLS: ct.Constant[int],
    N_BLOCKS_K: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    label_tile = ct.load(labels_ptr, index=(row,), shape=(1,))
    label = ct.reshape(label_tile, ())
    neg100 = -100
    is_valid = label != neg100
    safe_label = ct.where(is_valid, label, 0)

    # Online logsumexp over N_COLS in BLOCK_N chunks. Compute target logit
    # during the loop by masked reduction over each block.
    row_max = ct.full(shape=(), fill_value=-3.4e38, dtype=ct.float32)
    denom = ct.full(shape=(), fill_value=0.0, dtype=ct.float32)
    target = ct.full(shape=(), fill_value=0.0, dtype=ct.float32)
    neg_inf_tile = ct.full(
        shape=(1, BLOCK_N), fill_value=-3.4e38, dtype=ct.float32)
    safe_label_i32 = ct.astype(safe_label, ct.int32)
    for kb in range(N_BLOCKS_K):
        tile_bf = ct.load(
            logits_ptr, index=(row, kb), shape=(1, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        tile_f = ct.astype(tile_bf, ct.float32)
        ks = ct.arange(BLOCK_N, dtype=ct.int32) + kb * BLOCK_N
        in_bounds = ks < N_COLS
        in_bounds_2d = ct.reshape(in_bounds, (1, BLOCK_N))
        tile_masked = ct.where(in_bounds_2d, tile_f, neg_inf_tile)
        block_max = ct.max(tile_masked)
        new_max = ct.where(block_max > row_max, block_max, row_max)
        denom = denom * ct.exp(row_max - new_max)
        exp_tile = ct.exp(tile_f - new_max)
        zero_tile = ct.zeros((1, BLOCK_N), dtype=ct.float32)
        exp_masked = ct.where(in_bounds_2d, exp_tile, zero_tile)
        denom = denom + ct.sum(exp_masked)
        row_max = new_max
        # Target gather: sum(where(ks == safe_label, tile_f, 0)).
        match = ks == safe_label_i32
        picked = ct.where(
            ct.reshape(match, (1, BLOCK_N)), tile_f, zero_tile)
        target = target + ct.sum(picked)

    logp = target - row_max - ct.log(denom)
    loss_bf = ct.astype(logp, ct.bfloat16)
    loss_f = -ct.astype(loss_bf, ct.float32)

    loss_val = ct.where(is_valid, loss_f, ct.astype(0.0, ct.float32))
    valid_val = ct.where(is_valid, ct.astype(1.0, ct.float32),
                                       ct.astype(0.0, ct.float32))
    ct.store(loss_ptr, index=(row,),
             tile=ct.reshape(ct.astype(loss_val, ct.bfloat16), (1,)))
    ct.store(valid_ptr, index=(row,),
             tile=ct.reshape(valid_val, (1,)))


@ct.kernel
def _mean_kernel(
    loss_ptr, valid_ptr, out_ptr,
    BLOCK_N: ct.Constant[int],
):
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_N,),
                     padding_mode=ct.PaddingMode.ZERO)
    valids = ct.load(valid_ptr, index=(0,), shape=(BLOCK_N,),
                     padding_mode=ct.PaddingMode.ZERO)
    losses_f = ct.astype(losses, ct.float32)
    total_loss_bf = ct.astype(ct.sum(losses_f), ct.bfloat16)
    total_valid_bf = ct.astype(ct.sum(valids), ct.bfloat16)
    total_loss = ct.astype(total_loss_bf, ct.float32)
    total_valid = ct.astype(total_valid_bf, ct.float32)
    result = ct.astype(total_loss / total_valid, ct.bfloat16)
    ct.store(out_ptr, index=(0,), tile=ct.reshape(result, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="353f43da")
def oracle_forward(inputs):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_rows = int(matrix_shape[0])
    n_cols = int(matrix_shape[1])

    # No .contiguous() — pass the full logits tensor and mask in-kernel.
    logits_view = logits[:, :n_cols].view(view_shape)
    labels_1d = labels.view(-1)

    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.bfloat16)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.bfloat16)

    BLOCK_N = 8192
    n_blocks_k = (n_cols + BLOCK_N - 1) // BLOCK_N

    block_n_for_mean = 1
    while block_n_for_mean < n_rows:
        block_n_for_mean *= 2

    stream = torch.cuda.current_stream()
    ct.launch(stream, (n_rows, 1, 1), _xent_row_kernel,
              (labels_1d, logits, loss_per_row, valid_per_row,
               n_cols, n_blocks_k, BLOCK_N))
    ct.launch(stream, (1, 1, 1), _mean_kernel,
              (loss_per_row, valid_per_row, out.view(1), block_n_for_mean))

    return logits_view, out
