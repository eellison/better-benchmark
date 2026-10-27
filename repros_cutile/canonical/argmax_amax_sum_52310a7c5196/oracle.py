"""cuTile port of argmax_amax_sum_52310a7c5196: GPT-2 seq-classification tail.

Single-block kernel with BATCH=8, SEQ=1024, N_CLASSES=2.
- Argmax over positions with (token != 0) mask (via ``ct.max`` over ``where``).
- 2D gather of two bf16 logits per batch from logits[batch*SEQ + last_pos, c].
- 2-class log-softmax with bf16 round-trip (matches eager's convert_element_type
  bf16 cast then f32 promote).
- NLL loss with ignore-index masking, mean over valid.
- Side outputs: valid mask (b8[B,1]) and one-hot label equality (b8[B,2]).

The Triton oracle's ``add.rn.f32`` / ``sub.rn.f32`` / ``div.rn.f32`` inline PTX
is equivalent to cuTile's default RTNE ``+`` / ``-`` / ``/`` on ``float32`` —
no special handling needed.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _seqcls_tail_kernel(
    logits_ptr,        # bf16 [rows=8192, N_CLASSES=2]
    tokens_ptr,        # i64 [BATCH, SEQ]
    row_index_ptr,     # i64 [BATCH]
    labels_ptr,        # i64 [BATCH]
    argmax_ptr,        # i64 [BATCH]
    pooled_ptr,        # bf16 [BATCH, N_CLASSES]
    count_ptr,         # f32 [1]
    loss_ptr,          # f32 [1]
    valid_mask_ptr,    # b8 [BATCH]
    eq_ptr,            # b8 [BATCH, N_CLASSES]
    SEQ: ct.Constant[int],
    N_CLASSES: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    # --- argmax over (token != 0) positions -------------------------------
    tokens = ct.load(tokens_ptr, index=(0, 0), shape=(BLOCK_B, BLOCK_S))
    seq_1d = ct.arange(BLOCK_S, dtype=ct.int64)
    seq_2d = ct.reshape(seq_1d, (1, BLOCK_S))
    zero_i64_2d = ct.full((BLOCK_B, BLOCK_S), 0, dtype=ct.int64)
    positions = ct.where(tokens != 0, seq_2d, zero_i64_2d)
    last_pos = ct.max(positions, axis=1)  # (BLOCK_B,)
    ct.store(argmax_ptr, index=(0,), tile=last_pos)

    # --- 2D gather of pooled logits ---------------------------------------
    # The eager Repro views the flat [8192,2] bf16 tensor as [8,1024,2] and
    # then indexes with [row_index, argmax]. Flat-row = row_index * SEQ + argmax.
    row_index = ct.load(row_index_ptr, index=(0,), shape=(BLOCK_B,))
    flat_row = row_index * SEQ + last_pos  # (BLOCK_B,)
    c_idx = ct.arange(N_CLASSES, dtype=ct.int64)
    flat_row_2d = ct.reshape(flat_row, (BLOCK_B, 1))
    c_idx_2d = ct.reshape(c_idx, (1, N_CLASSES))
    pooled_bf16 = ct.gather(logits_ptr, (flat_row_2d, c_idx_2d))  # (B, 2)
    ct.store(pooled_ptr, index=(0, 0), tile=pooled_bf16)

    # --- 2-class log-softmax with bf16 round-trip -------------------------
    pooled_f32 = ct.astype(pooled_bf16, ct.float32)
    row_max = ct.max(pooled_f32, axis=1, keepdims=True)  # (B, 1)
    shifted = pooled_f32 - row_max                        # (B, 2)
    denom = ct.sum(ct.exp(shifted), axis=1, keepdims=True)  # (B, 1)
    log_denom = ct.log(denom)                             # (B, 1)
    log_prob = shifted - log_denom                        # (B, 2)
    log_prob_bf = ct.astype(log_prob, ct.bfloat16)
    log_prob_rt = ct.astype(log_prob_bf, ct.float32)      # (B, 2) after RT

    # --- NLL loss with ignore-index masking -------------------------------
    labels = ct.load(labels_ptr, index=(0,), shape=(BLOCK_B,))
    ignore_val_tile = ct.full((BLOCK_B,), -100, dtype=ct.int64)
    valid = labels != ignore_val_tile                    # (B,) bool
    zero_i64_1d = ct.full((BLOCK_B,), 0, dtype=ct.int64)
    safe_label = ct.where(valid, labels, zero_i64_1d)    # (B,) i64

    # target = log_prob_rt[b, safe_label[b]] — for N_CLASSES=2, expand to (B,2)
    # and pick the matching column via one-hot mask + sum.
    safe_label_2d = ct.reshape(safe_label, (BLOCK_B, 1))
    c_idx_2d_bcast = ct.reshape(c_idx, (1, N_CLASSES))
    onehot = safe_label_2d == c_idx_2d_bcast              # (B, 2) bool
    zero_f32_2d = ct.full((BLOCK_B, N_CLASSES), 0.0, dtype=ct.float32)
    target_expanded = ct.where(onehot, log_prob_rt, zero_f32_2d)
    target = ct.sum(target_expanded, axis=1)              # (B,)
    neg = 0.0 - target                                    # (B,)

    zero_f32_1d = ct.full((BLOCK_B,), 0.0, dtype=ct.float32)
    row_loss = ct.where(valid, neg, zero_f32_1d)          # (B,)
    ones_f32_1d = ct.full((BLOCK_B,), 1.0, dtype=ct.float32)
    valid_f32 = ct.where(valid, ones_f32_1d, zero_f32_1d)  # (B,)

    total_loss = ct.sum(row_loss)                         # scalar
    total_valid = ct.sum(valid_f32)                       # scalar
    ct.store(count_ptr, index=(0,), tile=ct.reshape(total_valid, (1,)))
    ct.store(loss_ptr, index=(0,),
             tile=ct.reshape(total_loss / total_valid, (1,)))

    # --- side outputs: valid mask (B,) and one-hot equality (B, N_CLASSES) -
    ct.store(valid_mask_ptr, index=(0,), tile=valid)

    # Eager's `eq` is `where(labels != -100, labels, 0) == [0,1]`, which is the
    # SAME as `onehot` above (both use `safe_label`).
    ct.store(eq_ptr, index=(0, 0), tile=onehot)


# d21e6ec7: (T([8192,2], bf16), T([8,1024], i64), T([8], i64, Index(8)),
#            T([8], i64, Index(2)), S([8,1024,2]), S([1,2]), S([8,2]))
@oracle_impl(hardware="B200", point="d21e6ec7", BLOCK_B=8, BLOCK_S=1024)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_S: int):
    logits, input_ids, row_index, labels, _s0, _s1, _s2 = inputs
    batch = int(labels.shape[0])
    seq = int(input_ids.shape[1])
    n_classes = 2

    argmax_out = torch.empty_strided(
        (batch,), (1,), device=logits.device, dtype=torch.int64)
    pooled = torch.empty_strided(
        (batch, n_classes), (n_classes, 1),
        device=logits.device, dtype=torch.bfloat16)
    count = torch.empty((), device=logits.device, dtype=torch.float32)
    loss = torch.empty((), device=logits.device, dtype=torch.float32)
    valid_mask = torch.empty((batch,), device=logits.device, dtype=torch.bool)
    eq = torch.empty_strided(
        (batch, n_classes), (n_classes, 1),
        device=logits.device, dtype=torch.bool)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _seqcls_tail_kernel,
        (
            logits,
            input_ids,
            row_index,
            labels,
            argmax_out,
            pooled,
            count.view(1),
            loss.view(1),
            valid_mask,
            eq,
            seq,
            n_classes,
            BLOCK_B,
            BLOCK_S,
        ),
    )
    # ne_2 in the eager path is [B, 1]; return valid_mask reshaped to match.
    return argmax_out, pooled, count, loss, valid_mask.view(batch, 1), eq
