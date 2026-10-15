"""cuTile port of argmax_amax_sum_ee1163128221: GPT-Neo last-token cross-entropy.

Computes last-nonzero-token argmax over input_ids, gathers logits, then
log-softmax over 2 classes, gathers label log-prob, computes mean loss.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _last_token_xent_kernel(
    labels_ptr,    # i64 [BATCH]
    logits_ptr,    # bf16 [BATCH*SEQ, 2]
    input_ids_ptr, # i64 [BATCH, SEQ]
    selected_ptr,  # bf16 [BATCH, 2]
    loss_ptr,      # bf16 [1]
    BATCH: ct.Constant[int],
    SEQ: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    tokens = ct.load(input_ids_ptr, index=(0, 0), shape=(BLOCK_B, BLOCK_N))
    positions = ct.arange(BLOCK_N, dtype=ct.int64)
    positions_2d = ct.reshape(positions, (1, BLOCK_N))
    # candidates = pos if token != 0 else 0
    candidates = ct.where(tokens != 0, positions_2d, ct.astype(positions_2d, ct.int64) * 0)
    last_pos = ct.max(candidates, axis=1)  # (BLOCK_B,)

    # row = batch * SEQ + last_pos; gather logits[row*2:row*2+2]
    batches = ct.arange(BLOCK_B, dtype=ct.int64)
    row_indices = batches * SEQ + last_pos  # (BLOCK_B,)
    row_indices_2d = ct.reshape(row_indices, (BLOCK_B, 1))
    col0 = ct.full(shape=(1, 1), fill_value=0, dtype=ct.int64)
    col1 = ct.full(shape=(1, 1), fill_value=1, dtype=ct.int64)
    logit0_bf16 = ct.gather(logits_ptr, (row_indices_2d, col0))
    logit1_bf16 = ct.gather(logits_ptr, (row_indices_2d, col1))
    # store to selected
    ct.scatter(selected_ptr, (batches, ct.full(shape=(BLOCK_B,), fill_value=0, dtype=ct.int64)),
               ct.reshape(logit0_bf16, (BLOCK_B,)))
    ct.scatter(selected_ptr, (batches, ct.full(shape=(BLOCK_B,), fill_value=1, dtype=ct.int64)),
               ct.reshape(logit1_bf16, (BLOCK_B,)))

    logit0 = ct.astype(ct.reshape(logit0_bf16, (BLOCK_B,)), ct.float32)
    logit1 = ct.astype(ct.reshape(logit1_bf16, (BLOCK_B,)), ct.float32)
    row_max = ct.where(logit0 > logit1, logit0, logit1)
    shifted0 = logit0 - row_max
    shifted1 = logit1 - row_max
    denom = ct.exp(shifted0) + ct.exp(shifted1)
    log_denom = ct.log(denom)
    log_prob0 = ct.astype(shifted0 - log_denom, ct.bfloat16)
    log_prob1 = ct.astype(shifted1 - log_denom, ct.bfloat16)

    labels = ct.load(labels_ptr, index=(0,), shape=(BLOCK_B,))
    valid = labels != -100
    safe_labels = ct.where(valid, labels, ct.full(shape=(BLOCK_B,), fill_value=0, dtype=ct.int64))
    gathered = ct.where(safe_labels == 0, log_prob0, log_prob1)
    nll = ct.astype(-ct.astype(gathered, ct.float32), ct.bfloat16)
    loss_terms = ct.where(valid, nll, ct.astype(ct.full(shape=(BLOCK_B,), fill_value=0.0, dtype=ct.float32), ct.bfloat16))
    loss_sum_bf16 = ct.astype(ct.sum(ct.astype(loss_terms, ct.float32)), ct.bfloat16)
    valid_count = ct.sum(ct.where(valid, ct.astype(ct.full(shape=(BLOCK_B,), fill_value=1.0, dtype=ct.float32), ct.float32),
                                   ct.astype(ct.full(shape=(BLOCK_B,), fill_value=0.0, dtype=ct.float32), ct.float32)))
    valid_count_bf16 = ct.astype(valid_count, ct.bfloat16)
    loss = ct.astype(loss_sum_bf16, ct.float32) / ct.astype(valid_count_bf16, ct.float32)
    ct.store(loss_ptr, index=(0,), tile=ct.reshape(ct.astype(loss, ct.bfloat16), (1,)))


@oracle_impl(hardware="B200", point="eacfffdd", BLOCK_B=32, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_N: int):
    labels, logits, input_ids, _shape_param_0 = inputs
    batch = int(labels.shape[0])
    seq = int(input_ids.shape[1])
    selected = torch.empty_strided(
        (batch, 2), (2, 1), device=logits.device, dtype=torch.bfloat16
    )
    loss = torch.empty((), device=logits.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _last_token_xent_kernel,
        (labels, logits, input_ids, selected, loss.view(1), batch, seq, BLOCK_B, BLOCK_N),
    )
    return selected, loss
