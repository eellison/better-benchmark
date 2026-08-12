"""cuTile port of argmax_amax_sum_a0354c147dd8 (NEW_PATTERN): GPT2 sequence
classification loss. Small (8,) rows and 2 classes. Handles last-token
argmax over the attention mask, log-softmax over the two classes, ignore-index
gather, and bf16 mean of losses."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ_LEN = 1024
N_CLASSES = 2


@ct.kernel
def _last_token_xent_kernel(
    labels_ptr,       # (8,) i64
    logits_ptr,       # (8, 1024, 2) bf16 view
    token_mask_ptr,   # (8, 1024) i64
    gathered_ptr,     # (8, 2) bf16
    loss_ptr,         # () bf16 scalar (via (1,) view)
    BLOCK_B: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    # Load labels (8,) and token mask (8, 1024)
    labels = ct.load(labels_ptr, index=(0,), shape=(BLOCK_B,))
    tokens = ct.load(token_mask_ptr, index=(0, 0), shape=(BLOCK_B, BLOCK_S))

    # For each batch b, find max position where tokens[b, pos] != 0
    positions = ct.arange(BLOCK_S, dtype=ct.int32)
    positions_2d = ct.reshape(positions, (1, BLOCK_S))
    ones_2d = ct.full(shape=(BLOCK_B, 1), fill_value=1, dtype=ct.int32)
    pos_b = ones_2d * positions_2d  # (B, S)
    zero = ct.full(shape=(BLOCK_B, BLOCK_S), fill_value=0, dtype=ct.int32)
    zero_i64 = ct.full(shape=(BLOCK_B, BLOCK_S), fill_value=0, dtype=ct.int64)
    is_nonzero = tokens != zero_i64
    candidate = ct.where(is_nonzero, pos_b, zero)
    last_positions = ct.max(candidate, axis=1)  # (B,)

    # Gather logits[b, last_positions[b], 0..1]
    batches = ct.astype(ct.arange(BLOCK_B, dtype=ct.int32), ct.int64)
    last_pos64 = ct.astype(last_positions, ct.int64)
    c0 = ct.zeros(shape=(BLOCK_B,), dtype=ct.int64)
    c1 = ct.full(shape=(BLOCK_B,), fill_value=1, dtype=ct.int64)
    logit0 = ct.gather(logits_ptr, (batches, last_pos64, c0))  # (B,) bf16
    logit1 = ct.gather(logits_ptr, (batches, last_pos64, c1))  # (B,) bf16

    # Store gathered [8, 2]
    logit0_2d = ct.reshape(logit0, (BLOCK_B, 1))
    logit1_2d = ct.reshape(logit1, (BLOCK_B, 1))
    # We need to store as (BLOCK_B, 2). Combine using two writes.
    # Instead, use a single (B, 2) tile by loading positions with cat.
    # Simpler: store into gathered as one dim=1 tile per column.
    ct.store(gathered_ptr, index=(0, 0), tile=ct.reshape(logit0_2d, (BLOCK_B, 1)))
    ct.store(gathered_ptr, index=(0, 1), tile=ct.reshape(logit1_2d, (BLOCK_B, 1)))

    # log-softmax: bf16 gathered -> f32
    l0 = ct.astype(logit0, ct.float32)
    l1 = ct.astype(logit1, ct.float32)
    row_max = ct.maximum(l0, l1)
    s0 = l0 - row_max
    s1 = l1 - row_max
    denom = ct.exp(s0) + ct.exp(s1)
    log_denom = ct.log(denom)
    # bf16 log-softmax then re-promote
    lsm0_bf = ct.astype(s0 - log_denom, ct.bfloat16)
    lsm1_bf = ct.astype(s1 - log_denom, ct.bfloat16)
    lsm0 = ct.astype(lsm0_bf, ct.float32)
    lsm1 = ct.astype(lsm1_bf, ct.float32)

    # Select target and negate
    label_i32 = ct.astype(labels, ct.int32)
    ignore_mask = labels != ct.full(shape=(BLOCK_B,), fill_value=-100, dtype=ct.int64)
    safe_label = ct.where(
        ignore_mask, label_i32,
        ct.full(shape=(BLOCK_B,), fill_value=0, dtype=ct.int32),
    )
    # target = safe_label==0 ? lsm0 : lsm1
    is_zero_label = safe_label == ct.full(shape=(BLOCK_B,), fill_value=0, dtype=ct.int32)
    target = ct.where(is_zero_label, lsm0, lsm1)
    neg = -target
    # Cast to bf16 then f32 to match compiled path
    neg_bf = ct.astype(neg, ct.bfloat16)
    zero_bf = ct.full(shape=(BLOCK_B,), fill_value=0.0, dtype=ct.bfloat16)
    loss_bf = ct.where(ignore_mask, neg_bf, zero_bf)

    valid_bf = ct.astype(
        ct.astype(
            ct.where(
                ignore_mask,
                ct.full(shape=(BLOCK_B,), fill_value=1, dtype=ct.int32),
                ct.full(shape=(BLOCK_B,), fill_value=0, dtype=ct.int32),
            ),
            ct.int64,
        ),
        ct.bfloat16,
    )
    # sum(loss) / valid_count in bf16
    total_loss = ct.sum(ct.astype(loss_bf, ct.float32))
    total_valid = ct.sum(ct.astype(valid_bf, ct.float32))
    result = ct.astype(total_loss, ct.bfloat16) / ct.astype(total_valid, ct.bfloat16)
    ct.store(loss_ptr, index=(0,), tile=ct.reshape(result, (1,)))


@oracle_impl(hardware="B200", point="91cac6df", BLOCK_B=8, BLOCK_S=1024)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_S: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs

    # index = gather from view [8, 1024, 2] using [iota, argmax]
    gathered = torch.empty_strided(
        (BATCH, N_CLASSES),
        (N_CLASSES, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    loss = torch.empty((), device=arg1_1.device, dtype=torch.bfloat16)

    view3 = arg1_1.view(BATCH, SEQ_LEN, N_CLASSES)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _last_token_xent_kernel,
        (arg0_1, view3, arg2_1, gathered, loss.view(1), BLOCK_B, BLOCK_S),
    )
    return (gathered, loss)
