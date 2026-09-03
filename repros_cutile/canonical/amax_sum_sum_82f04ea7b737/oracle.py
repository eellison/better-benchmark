"""cuTile port of amax_sum_sum_82f04ea7b737: Blenderbot biased cross-entropy.

Per-row kernel:
* Load bf16 logits row (padded to 16384) + f32 bias row.
* Compute add = logits.f32 + bias (padded lanes are 0).
* Emit `add` via scatter (masked to valid cols).
* Compute row max and log-sum-exp with padded lanes masked to -inf.

Loss and mean are computed in torch on the returned `add`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 8008
N_COLS_PAD = 8192
NEG_INF = float("-inf")


@ct.kernel
def _biased_xent_row_kernel(
    logits_ptr,   # bf16 [rows, N_COLS]
    bias_ptr,     # f32  [N_COLS]
    add_ptr,      # f32  [rows, N_COLS]
    amax_ptr,     # f32  [rows]
    log_ptr,      # f32  [rows]
    N_COLS_C: ct.Constant[int],
    N_COLS_PAD_C: ct.Constant[int],
):
    row = ct.bid(0)
    logits = ct.load(logits_ptr, index=(row, 0), shape=(1, N_COLS_PAD_C),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(N_COLS_PAD_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    bias_2d = ct.reshape(bias, (1, N_COLS_PAD_C))
    add = ct.astype(logits, ct.float32) + bias_2d

    cols = ct.arange(N_COLS_PAD_C, dtype=ct.int32)
    valid = cols < N_COLS_C
    r_idx = ct.full(shape=(N_COLS_PAD_C,), fill_value=row, dtype=ct.int32)
    ct.scatter(add_ptr, (r_idx, cols), ct.reshape(add, (N_COLS_PAD_C,)), mask=valid)

    valid_2d = ct.reshape(valid, (1, N_COLS_PAD_C))
    neg_inf_2d = ct.full(shape=(1, N_COLS_PAD_C), fill_value=NEG_INF, dtype=ct.float32)
    add_for_max = ct.where(valid_2d, add, neg_inf_2d)
    row_max = ct.max(add_for_max)
    exp_arg = add - row_max
    exp_v = ct.exp(exp_arg)
    zero_2d = ct.full(shape=(1, N_COLS_PAD_C), fill_value=0.0, dtype=ct.float32)
    exp_masked = ct.where(valid_2d, exp_v, zero_2d)
    denom = ct.sum(exp_masked)
    log_denom = ct.log(denom)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(log_ptr, index=(row,), tile=ct.reshape(log_denom, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="a0e86354")
def oracle_forward(inputs):
    logits, bias, labels, shape0, shape1 = inputs
    device = logits.device
    add_shape = _shape_tuple(shape0)
    n_rows = int(labels.numel())

    add = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=device,
        dtype=torch.float32,
    )
    add_2d = add.view(n_rows, N_COLS)
    amax = torch.empty_strided((n_rows, 1), (1, 1), device=device, dtype=torch.float32)
    amax_1d = amax.view(n_rows)
    log = torch.empty_strided((n_rows, 1), (1, 1), device=device, dtype=torch.float32)
    log_1d = log.view(n_rows)

    bias_1d = bias.view(N_COLS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _biased_xent_row_kernel,
        (logits, bias_1d, add_2d, amax_1d, log_1d, N_COLS, N_COLS_PAD),
    )

    view_2 = labels.view(-1)
    ne_mask = view_2 != -100
    safe_labels = torch.where(ne_mask, view_2, torch.zeros_like(view_2))
    add_target = add_2d.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    logp = add_target - amax_1d - log_1d
    neg_logp = -logp
    zero_scalar = torch.zeros((), device=device, dtype=torch.float32)
    where_1 = torch.where(ne_mask, neg_logp, zero_scalar)
    count = ne_mask.sum().to(torch.float32)
    total_loss = where_1.sum()
    div = total_loss / count

    return add, amax, log, count, div
