"""cuTile port of sum_f78ec7cb2b81 (ALGEBRAIC_ELIMINATION): Blenderbot XE
backward. Per-row: guarded label-scalar sum replaces the one-hot reduction,
then per-element `incoming + one_hot_scaled - exp(shifted) * row_sum`
in fp32 with bf16 rounding on the final store.

vocab=8008 (not power of 2). Pad columns to 8192, compute, then slice.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 2048
VOCAB = 8008
VOCAB_PAD = 8192  # next pow2


@ct.kernel
def _one_hot_exp_bf16_kernel(
    numerator_ptr,     # f32 [1]
    denominator_ptr,   # f32 [1]
    labels_ptr,        # i64 [ROWS]
    logits_ptr,        # f32 [ROWS, VOCAB_PAD]
    row_shift0_ptr,    # f32 [ROWS]
    row_shift1_ptr,    # f32 [ROWS]
    incoming_grad_ptr, # f32 [ROWS, VOCAB_PAD]
    out_ptr,           # bf16 [ROWS, VOCAB_PAD]
    VOCAB_N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    cols = ct.arange(BLOCK_N, dtype=ct.int32) + col_block * BLOCK_N

    raw_label = ct.load(labels_ptr, index=(row,), shape=(1,))
    active = raw_label != -100
    selected_label = ct.where(active, raw_label, ct.zeros((1,), dtype=ct.int64))
    scale_num = ct.astype(
        ct.load(numerator_ptr, index=(0,), shape=(1,)), ct.float32
    )
    scale_den = ct.astype(
        ct.load(denominator_ptr, index=(0,), shape=(1,)), ct.float32
    )
    scale = scale_num / scale_den

    active_f = ct.astype(active, ct.float32)
    target_in_vocab = active & (selected_label >= 0) & (selected_label < VOCAB_N)
    tiv_f = ct.astype(target_in_vocab, ct.float32)

    scale_delta = scale - scale
    finite_row_sum = tiv_f * (-scale)
    scale_is_finite = scale_delta == ct.zeros((1,), dtype=ct.float32)
    sif_f = ct.astype(scale_is_finite, ct.float32)
    active_row_sum = sif_f * finite_row_sum + (1.0 - sif_f) * scale_delta
    row_sum = active_f * active_row_sum

    row_scale = active_f * scale

    row_sum_2d = ct.reshape(row_sum, (1, 1))
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    selected_label_2d = ct.reshape(selected_label, (1, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))

    matches = selected_label_2d == ct.astype(cols_2d, ct.int64)
    matches_f = ct.astype(matches, ct.float32)
    one_hot = -matches_f
    one_hot_scaled = one_hot * row_scale_2d

    incoming = ct.astype(
        ct.load(incoming_grad_ptr, index=(row, col_block), shape=(1, BLOCK_N)),
        ct.float32,
    )
    logits = ct.astype(
        ct.load(logits_ptr, index=(row, col_block), shape=(1, BLOCK_N)),
        ct.float32,
    )
    shift0 = ct.astype(
        ct.load(row_shift0_ptr, index=(row,), shape=(1,)),
        ct.float32,
    )
    shift1 = ct.astype(
        ct.load(row_shift1_ptr, index=(row,), shape=(1,)),
        ct.float32,
    )
    shift0_2d = ct.reshape(shift0, (1, 1))
    shift1_2d = ct.reshape(shift1, (1, 1))

    shifted = logits - shift0_2d - shift1_2d
    exp_values = ct.exp(shifted)
    exp_times_sum = exp_values * row_sum_2d
    delta = one_hot_scaled - exp_times_sum
    out = ct.astype(incoming + delta, ct.bfloat16)
    ct.store(out_ptr, index=(row, col_block), tile=out)


@oracle_impl(hardware="B200", point="3cca3d26", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    numerator, denominator, labels, logits, row_shift0, row_shift1, incoming_grad, *_ = inputs

    device = logits.device
    labels_flat = labels.view(-1)  # (2048,)
    logits_flat = logits.view(ROWS, VOCAB)  # (2048, 8008)
    incoming_flat = incoming_grad.view(ROWS, VOCAB)

    logits_padded = torch.zeros((ROWS, VOCAB_PAD), device=device, dtype=torch.float32)
    logits_padded[:, :VOCAB] = logits_flat
    incoming_padded = torch.zeros((ROWS, VOCAB_PAD), device=device, dtype=torch.float32)
    incoming_padded[:, :VOCAB] = incoming_flat

    row_shift0_flat = row_shift0.view(ROWS)
    row_shift1_flat = row_shift1.view(ROWS)
    num_flat = numerator.view(1)
    den_flat = denominator.view(1)

    out_padded = torch.empty(
        (ROWS, VOCAB_PAD), device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, ct.cdiv(VOCAB_PAD, BLOCK_N), 1),
        _one_hot_exp_bf16_kernel,
        (num_flat, den_flat, labels_flat, logits_padded, row_shift0_flat,
         row_shift1_flat, incoming_padded, out_padded, VOCAB, BLOCK_N),
    )
    out = out_padded[:, :VOCAB].contiguous()
    return out, torch.as_strided(out, (VOCAB, ROWS), (1, VOCAB))
