"""cuTile port of amax_sum_sum_3f000d9caa57: sliced-vocab cross-entropy.

Input: bf16[rows, 50272] sliced to first 50265 columns.
Per row: online logsumexp over 50265 cols (streamed in BLOCK_N chunks).
Compute row_max, log_denom. Then Python-side: gather target column, apply
bf16 log-softmax rounding, mask by ignore index, return scalar mean loss.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS_SLICED = 50265


@ct.kernel
def _lse_kernel(
    logits_ptr,     # bf16 (rows, N_COLS_FULL)
    amax_out_ptr,   # f32 (rows,)
    log_out_ptr,    # f32 (rows,)
    N_COLS: ct.Constant[int],
    N_BLOCKS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    row_max = ct.full((1,), float("-inf"), dtype=ct.float32)
    denom = ct.zeros((1,), dtype=ct.float32)

    for block_idx in ct.static_iter(range(N_BLOCKS)):
        logits = ct.load(logits_ptr, index=(row, block_idx), shape=(1, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        cols = block_idx * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
        valid = cols < N_COLS
        logits_1d = ct.reshape(logits, (BLOCK_N,))
        logits_f = ct.astype(logits_1d, ct.float32)
        neg_inf = ct.full((BLOCK_N,), float("-inf"), dtype=ct.float32)
        logits_masked = ct.where(valid, logits_f, neg_inf)
        block_max = ct.max(logits_masked, keepdims=True)
        next_max = ct.where(row_max > block_max, row_max, block_max)
        next_max_bc = ct.reshape(next_max, (1,))
        shift = logits_masked - next_max_bc
        exp_shift = ct.exp(shift)
        zero_bn = ct.zeros((BLOCK_N,), dtype=ct.float32)
        exp_masked = ct.where(valid, exp_shift, zero_bn)
        block_sum = ct.sum(exp_masked, keepdims=True)
        row_shift = ct.exp(row_max - next_max)
        denom = denom * row_shift + block_sum
        row_max = next_max

    log_denom = ct.log(denom)
    ct.store(amax_out_ptr, index=(row,), tile=row_max)
    ct.store(log_out_ptr, index=(row,), tile=log_denom)


def _view_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="4445581e", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="f78f1ecd", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="e195ea0d", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = _view_shape(shape_3d)
    matrix_shape = _view_shape(shape_2d)
    n_rows = int(matrix_shape[0])
    n_cols = int(matrix_shape[1])

    logits_view = logits[:, :n_cols].view(view_shape)
    labels_1d = labels.view(-1)

    amax = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)

    N_BLOCKS = (n_cols + BLOCK_N - 1) // BLOCK_N
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _lse_kernel,
        (logits, amax, log, n_cols, N_BLOCKS, BLOCK_N),
    )

    # Python-side epilogue
    valid = labels_1d != -100
    safe_labels = torch.where(valid, labels_1d, torch.zeros_like(labels_1d))
    # Gather target column from sliced logits view
    logits_sliced_f32 = logits[:, :n_cols].to(torch.float32)
    target = torch.gather(logits_sliced_f32, 1, safe_labels.unsqueeze(1)).squeeze(1)
    # log-softmax
    logp = target - amax - log  # f32
    # bf16 rounding boundary
    logp_bf16_f32 = logp.to(torch.bfloat16).to(torch.float32)
    loss = -logp_bf16_f32
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    count = valid.to(torch.float32).sum()
    div = loss.sum() / count

    amax_r = amax.view(n_rows, 1)
    log_r = log.view(n_rows, 1)
    return logits_view, amax_r, log_r, count, div
