"""cuTile port of amax_sum_sum_04ddf882ff17: MobileBERT biased log-softmax + cross entropy.

Uses cuTile for the per-row biased-add + log-softmax kernel over 30522 columns
(rounded up to 32768 with -inf padding), then computes loss and count via torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 30522
N_COLS_PAD = 32768  # next power of 2


@ct.kernel
def _biased_logsoftmax_row_kernel(
    logits_ptr,      # bf16 [rows, LOGITS_STRIDE=30528] flat view
    bias_ptr,        # f32  [N_COLS]
    biased_ptr,      # bf16 [rows, N_COLS]
    logp_ptr,        # bf16 [rows, N_COLS]
    N_COLS: ct.Constant[int],
    N_COLS_PAD: ct.Constant[int],
):
    row = ct.bid(0)
    # Load full logits row up to N_COLS_PAD (OOB = 0 since we use padding).
    # Then mask out beyond N_COLS.
    logits_bf = ct.load(
        logits_ptr, index=(row, 0), shape=(1, N_COLS_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias_bf = ct.load(
        bias_ptr, index=(0,), shape=(N_COLS_PAD,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits_f = ct.astype(logits_bf, ct.float32)
    bias_f = ct.astype(bias_bf, ct.float32)
    bias_2d = ct.reshape(bias_f, (1, N_COLS_PAD))
    biased_f = logits_f + bias_2d
    biased_bf = ct.astype(biased_f, ct.bfloat16)

    cols = ct.arange(N_COLS_PAD, dtype=ct.int32)
    valid = cols < N_COLS
    valid_2d = ct.reshape(valid, (1, N_COLS_PAD))

    # Store biased with mask - use zero for invalid positions since we only
    # write to the biased [rows, N_COLS] tensor which has exactly N_COLS cols.
    # But cuTile only supports whole-tile stores.
    # Approach: pad the biased tensor storage to N_COLS_PAD and store the whole tile.

    ct.store(biased_ptr, index=(row, 0), tile=biased_bf)

    # Log-softmax: for reduction we need -inf where OOB.
    neg_inf_2d = ct.full((1, N_COLS_PAD), float("-inf"), dtype=ct.float32)
    biased_for_reduce = ct.where(valid_2d, ct.astype(biased_bf, ct.float32), neg_inf_2d)
    row_max = ct.max(biased_for_reduce, axis=1, keepdims=True)
    shifted = biased_for_reduce - row_max
    numer = ct.exp(shifted)
    zero_f = ct.full((1, N_COLS_PAD), 0.0, dtype=ct.float32)
    numer_masked = ct.where(valid_2d, numer, zero_f)
    denom = ct.sum(numer_masked, axis=1, keepdims=True)
    log_denom = ct.log(denom)
    logp = ct.astype(biased_bf, ct.float32) - row_max - log_denom
    logp_bf = ct.astype(logp, ct.bfloat16)
    ct.store(logp_ptr, index=(row, 0), tile=logp_bf)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="8f164373")
def oracle_forward(inputs):
    logits, bias, labels, shape_3d, shape_2d, _output_shape = inputs
    biased_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_rows = int(matrix_shape[0])
    n_cols = int(matrix_shape[1])
    device = logits.device

    # Padded output tensors for cuTile writes.
    biased_pad = torch.empty((n_rows, N_COLS_PAD), device=device, dtype=torch.bfloat16)
    logp_pad = torch.empty((n_rows, N_COLS_PAD), device=device, dtype=torch.bfloat16)

    # logits shape [32768, 30528], we take slice [:, :30522].
    # cuTile: read the [rows, 30528] flat logits and mask beyond 30522.
    # But logits ptr should be a 2D view [rows, 30528].
    logits_2d = logits  # already [32768, 30528]

    # bias: 1D [30522] — cuTile OOB padding to N_COLS_PAD.
    bias_1d = bias

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _biased_logsoftmax_row_kernel,
        (logits_2d, bias_1d, biased_pad, logp_pad,
         n_cols, N_COLS_PAD),
    )

    # Slice back to N_COLS.
    biased = biased_pad[:, :n_cols].contiguous()
    logp = logp_pad[:, :n_cols].contiguous()

    # Compute count + loss via torch.
    labels_flat = labels.view(-1)
    ne = labels_flat != -100
    safe_labels = torch.where(ne, labels_flat, torch.zeros_like(labels_flat))
    unsqueezed = safe_labels.unsqueeze(1)
    logp_f = logp.to(torch.float32)
    gathered = torch.gather(logp_f, 1, unsqueezed).squeeze(1)
    losses = -gathered
    losses_masked = torch.where(ne, losses, torch.zeros_like(losses))
    count = ne.to(torch.float32).sum()
    loss_total = losses_masked.sum()
    div = loss_total / count

    # Return biased_view_3d, logsoftmax, count, div.
    biased_view = biased.view(biased_shape)
    return biased_view, logp, count, div
