"""cuTile port of sum_89a9de6a6901: MT5 cross-entropy backward.

VOCAB=250112 which factors as 2^7 * 1954 — no power-of-2 BLOCK_N above 128
divides it. We tile with BLOCK_N=128 (grid=1954, exact partitioning). ROWS=4096
divides BLOCK_M=8 (grid=512).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 4096
VOCAB = 250112
OUT_SHAPE = (ROWS, VOCAB)
OUT_STRIDE = (VOCAB, 1)
PERMUTE_SHAPE = (VOCAB, ROWS)
PERMUTE_STRIDE = (1, VOCAB)


@ct.kernel
def _mt5_loss_backward_kernel(
    numerator_ptr,   # f32 [1]
    denominator_ptr, # f32 [1]
    labels_ptr,      # i64 [ROWS]
    logits_ptr,      # bf16 [ROWS, VOCAB]
    shift0_ptr,      # f32 [ROWS]
    shift1_ptr,      # f32 [ROWS]
    residual_ptr,    # bf16 [ROWS, VOCAB]
    out_ptr,         # bf16 [ROWS, VOCAB]
    VOCAB_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    numerator = ct.load(numerator_ptr, index=(0,), shape=(1,))
    denominator = ct.load(denominator_ptr, index=(0,), shape=(1,))
    scale_val = numerator / denominator  # tile of shape (1,)

    label = ct.load(labels_ptr, index=(pid_m,), shape=(BLOCK_M,))
    label_f32 = ct.astype(label, ct.float32)
    active = label != ct.full((BLOCK_M,), -100, dtype=ct.int64)
    zero_i64 = ct.full((BLOCK_M,), 0, dtype=ct.int64)
    safe_label = ct.where(active, label, zero_i64)
    in_vocab = (safe_label >= zero_i64) & (safe_label < ct.full((BLOCK_M,), VOCAB_, dtype=ct.int64))

    # Broadcast scale.
    zero_f = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    scale_broadcast = zero_f + ct.reshape(scale_val, (1,))
    row_scale = ct.where(active, scale_broadcast, zero_f)

    # One-hot label gradient: (BLOCK_M, BLOCK_N) — -1 where col matches safe_label.
    cols = ct.arange(BLOCK_N, dtype=ct.int64) + ct.full((BLOCK_N,), pid_n * BLOCK_N, dtype=ct.int64)
    safe_label_2d = ct.reshape(safe_label, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    match = safe_label_2d == cols_2d
    one_hot = ct.where(
        match,
        ct.full((BLOCK_M, BLOCK_N), -1.0, dtype=ct.float32),
        ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32),
    )
    row_scale_2d = ct.reshape(row_scale, (BLOCK_M, 1))
    label_grad = ct.astype(
        ct.astype(one_hot * row_scale_2d, ct.bfloat16), ct.float32
    )

    # row_sum for the exp * (-scale) term when in_vocab and scale finite:
    scale_val_broadcast = zero_f + ct.reshape(scale_val, (1,))
    scale_delta = scale_val_broadcast - scale_val_broadcast
    scale_is_finite = scale_delta == zero_f
    rounded_neg_scale = ct.astype(
        ct.astype(-scale_val_broadcast, ct.bfloat16), ct.float32
    )
    finite_row_sum = ct.where(in_vocab, rounded_neg_scale, zero_f)
    active_row_sum = ct.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = ct.where(active, active_row_sum, zero_f)

    logits = ct.astype(
        ct.load(logits_ptr, index=(pid_m, pid_n), shape=(BLOCK_M, BLOCK_N)),
        ct.float32,
    )
    shift0 = ct.load(shift0_ptr, index=(pid_m,), shape=(BLOCK_M,))
    shift1 = ct.load(shift1_ptr, index=(pid_m,), shape=(BLOCK_M,))
    shift0_2d = ct.reshape(shift0, (BLOCK_M, 1))
    shift1_2d = ct.reshape(shift1, (BLOCK_M, 1))
    shifted = ct.astype(
        ct.astype(logits - shift0_2d - shift1_2d, ct.bfloat16), ct.float32
    )
    exp_shifted = ct.exp(shifted)

    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    exp_times_sum = exp_shifted * row_sum_2d
    delta = ct.astype(label_grad - exp_times_sum, ct.bfloat16)
    residual = ct.load(residual_ptr, index=(pid_m, pid_n), shape=(BLOCK_M, BLOCK_N))
    out = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(delta, ct.float32),
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(pid_m, pid_n), tile=out)


@oracle_impl(hardware="B200", point="55df3967", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    (
        numerator, denominator, labels, logits, shift0, shift1, residual,
        _shape0, _shape1, _shape2, _shape3, _shape4,
    ) = inputs

    zero = torch.zeros((), device=logits.device, dtype=torch.float32)
    out = torch.empty_strided(
        OUT_SHAPE, OUT_STRIDE,
        device=logits.device, dtype=torch.bfloat16,
    )

    assert VOCAB % BLOCK_N == 0, f"BLOCK_N must divide VOCAB={VOCAB}"
    assert ROWS % BLOCK_M == 0, f"BLOCK_M must divide ROWS={ROWS}"
    grid_m = ROWS // BLOCK_M
    grid_n = VOCAB // BLOCK_N

    labels_flat = labels.view(ROWS)
    numerator_1d = numerator.reshape(1)
    denominator_1d = denominator.reshape(1)
    shift0_1d = shift0.view(ROWS)
    shift1_1d = shift1.view(ROWS)
    logits_2d = logits.view(ROWS, VOCAB)
    residual_2d = residual.view(ROWS, VOCAB)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (grid_m, grid_n, 1),
        _mt5_loss_backward_kernel,
        (numerator_1d, denominator_1d, labels_flat, logits_2d, shift0_1d, shift1_1d,
         residual_2d, out, VOCAB, BLOCK_M, BLOCK_N),
    )

    return zero, out, out.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE)
