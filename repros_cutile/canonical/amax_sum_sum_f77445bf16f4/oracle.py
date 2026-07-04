"""cuTile port of amax_sum_sum_f77445bf16f4: GoogleFnet f32 ignore-index cross-entropy.

Ports the Triton `_xent_stats_rows_kernel` + `_mean_reduce_kernel`. The cuTile
kernel computes per-row (amax, log_denom) via chunked online reduction over the
non-power-of-2 32000-column vocabulary. Python epilogue does target gather,
ignore mask, and mean division.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_logsumexp_kernel(
    logits_ptr,   # f32 [n_rows, padded_cols]
    amax_ptr,     # f32 [n_rows]
    log_ptr,      # f32 [n_rows]
    N_COLS: ct.Constant[int],
    N_BLOCKS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    row_max = ct.full((1,), float("-inf"), dtype=ct.float32)
    denom = ct.zeros((1,), dtype=ct.float32)
    for block_idx in ct.static_iter(range(N_BLOCKS)):
        cols = block_idx * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
        valid = cols < N_COLS
        logits = ct.load(
            logits_ptr,
            index=(row, block_idx),
            shape=(1, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        logits_1d = ct.reshape(logits, (BLOCK_N,))
        neg_inf = ct.full((BLOCK_N,), float("-inf"), dtype=ct.float32)
        logits_masked = ct.where(valid, logits_1d, neg_inf)

        block_max = ct.max(logits_masked, keepdims=True)
        new_max = ct.where(row_max > block_max, row_max, block_max)
        new_max_bc = ct.reshape(new_max, (1,))
        shift = logits_masked - new_max_bc
        exp_shift = ct.exp(shift)
        zero_bn = ct.zeros((BLOCK_N,), dtype=ct.float32)
        exp_shift_masked = ct.where(valid, exp_shift, zero_bn)
        block_sum = ct.sum(exp_shift_masked, keepdims=True)
        row_shift = ct.exp(row_max - new_max)
        denom = denom * row_shift + block_sum
        row_max = new_max

    ct.store(amax_ptr, index=(row,), tile=row_max)
    ct.store(log_ptr, index=(row,), tile=ct.log(denom))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a49a430b", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, labels, shape_3d, shape_2d = inputs
    logits_view = logits.view(_shape_tuple(shape_3d))
    logits_2d = logits_view.view(_shape_tuple(shape_2d))
    labels_1d = labels.view(-1)
    n_rows = int(logits_2d.shape[0])
    n_cols = int(logits_2d.shape[1])
    device = logits.device

    # Pad logits to next power-of-2 multiple of BLOCK_N.
    n_blocks = (n_cols + BLOCK_N - 1) // BLOCK_N
    padded_cols = n_blocks * BLOCK_N
    logits_padded = torch.zeros(
        (n_rows, padded_cols), device=device, dtype=torch.float32,
    )
    logits_padded[:, :n_cols] = logits_2d

    amax_1d = torch.empty((n_rows,), device=device, dtype=torch.float32)
    log_1d = torch.empty((n_rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _row_logsumexp_kernel,
        (logits_padded, amax_1d, log_1d, n_cols, n_blocks, BLOCK_N),
    )

    amax = amax_1d.view(n_rows, 1)
    log = log_1d.view(n_rows, 1)

    # Python-side epilogue: target gather, ignore mask, mean division.
    valid = labels_1d != -100
    safe_labels = torch.where(valid, labels_1d, torch.zeros_like(labels_1d))
    target_f32 = torch.gather(logits_2d, 1, safe_labels.unsqueeze(1)).squeeze(1)
    # logp = target - amax_1d - log_1d
    logp = (target_f32 - amax_1d) - log_1d
    loss = -logp
    loss_masked = torch.where(valid, loss, torch.zeros_like(loss))

    count = valid.to(torch.float32).sum()
    total_loss = loss_masked.sum()
    div = total_loss / count

    return logits_view, amax, log, count, div
