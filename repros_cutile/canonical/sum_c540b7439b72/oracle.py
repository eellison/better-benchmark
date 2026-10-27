"""cuTile port of sum_c540b7439b72: Blenderbot causal-LM CE-backward dense grad.

Ports the Triton `_one_hot_exp_add_kernel` — for every element of a
[ROWS=4096, VOCAB=8008] grid, computes the CE-backward gradient given
per-row shifts and a per-row (label, scale) pair, one-hot masked, then
adds an incoming gradient. cuTile fp32 arithmetic is already round-to-
nearest-even, so the Triton PTX helpers become plain operators. VOCAB=8008
is non-pow2 but ROWS*VOCAB is divisible by BLOCK=1024, so we tile over the
flat 1D storage and derive `(row, col)` via `flat_idx // VOCAB`, `% VOCAB`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ         # 4096
VOCAB = 8008
NUMEL = ROWS * VOCAB       # 32,800,768 -> divisible by 1024


@ct.kernel
def _one_hot_exp_add_flat_kernel(
    numerator_ptr,        # f32[1]
    denominator_ptr,      # f32[1]
    labels_ptr,           # i64[ROWS]
    logits_ptr,           # bf16[NUMEL]
    row_shift0_ptr,       # f32[ROWS]
    row_shift1_ptr,       # f32[ROWS]
    incoming_grad_ptr,    # bf16[NUMEL]
    out_ptr,              # bf16[NUMEL]
    VOCAB_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    row = idx // VOCAB_
    col = idx - row * VOCAB_  # equivalent to idx % VOCAB_

    row_i64 = ct.astype(row, ct.int64)
    raw_label = ct.gather(labels_ptr, row_i64)
    valid_label = raw_label != -100
    zero_i64 = ct.zeros((BLOCK,), dtype=ct.int64)
    safe_label = ct.where(valid_label, raw_label, zero_i64)

    numerator = ct.astype(
        ct.load(numerator_ptr, index=(0,), shape=(1,)), ct.float32
    )
    denominator = ct.astype(
        ct.load(denominator_ptr, index=(0,), shape=(1,)), ct.float32
    )
    scale = numerator / denominator                       # scalar tile shape (1,)
    scale_delta = scale - scale                           # 0.0 (finite) or nan (inf)
    scale_is_finite = scale_delta == 0.0

    # rounded_target_sum = round_bf16(-scale)
    neg_scale = -scale                                    # shape (1,)
    rounded_target_sum = ct.astype(ct.astype(neg_scale, ct.bfloat16), ct.float32)
    # For each row: if valid_label, use rounded_target_sum else 0.
    zero_f = ct.astype(ct.zeros((BLOCK,), dtype=ct.float32), ct.float32)
    finite_row_sum = ct.where(valid_label, rounded_target_sum, zero_f)
    active_row_sum = ct.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = ct.where(valid_label, active_row_sum, zero_f)

    row_scale = ct.where(valid_label, scale, zero_f)

    # one_hot per element: -1 if col == safe_label, else 0
    col_i64 = ct.astype(col, ct.int64)
    one_hot_bool = safe_label == col_i64
    neg_one_f = ct.astype(ct.full((BLOCK,), -1.0, dtype=ct.float32), ct.float32)
    one_hot_val = ct.where(one_hot_bool, neg_one_f, zero_f)
    one_hot_scaled = ct.astype(
        ct.astype(one_hot_val * row_scale, ct.bfloat16), ct.float32
    )

    # Load logits/incoming/shifts
    logits = ct.astype(
        ct.load(logits_ptr, index=(pid,), shape=(BLOCK,)), ct.float32
    )
    incoming = ct.astype(
        ct.load(incoming_grad_ptr, index=(pid,), shape=(BLOCK,)), ct.float32
    )
    shift0 = ct.gather(row_shift0_ptr, row_i64)
    shift1 = ct.gather(row_shift1_ptr, row_i64)

    shifted_bf16 = ct.astype(logits - shift0 - shift1, ct.bfloat16)
    shifted_f = ct.astype(shifted_bf16, ct.float32)
    exp_val = ct.exp(shifted_f)
    exp_times_sum = exp_val * row_sum
    delta_bf16 = ct.astype(one_hot_scaled - exp_times_sum, ct.bfloat16)
    delta_f = ct.astype(delta_bf16, ct.float32)
    out = ct.astype(incoming + delta_f, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="bb71d599", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        incoming_grad,
        *_shape,
    ) = inputs

    out = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )

    numerator_flat = numerator.view(1)
    denominator_flat = denominator.view(1)
    labels_flat = labels.reshape(-1)      # (ROWS,) i64
    logits_flat = logits.reshape(NUMEL)   # (NUMEL,) bf16
    incoming_flat = incoming_grad.reshape(NUMEL)
    row_shift0_flat = row_shift0.view(ROWS)
    row_shift1_flat = row_shift1.view(ROWS)
    out_flat = out.view(NUMEL)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK), 1, 1),
        _one_hot_exp_add_flat_kernel,
        (
            numerator_flat, denominator_flat, labels_flat,
            logits_flat, row_shift0_flat, row_shift1_flat,
            incoming_flat, out_flat, VOCAB, BLOCK,
        ),
    )
    return out, torch.as_strided(out, (VOCAB, ROWS), (1, VOCAB))
