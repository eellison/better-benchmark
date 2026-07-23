"""cuTile port of argmax_amax_sum_6fdd39a1aff9: GPT-Neo sequence-classification tail.

Single-block kernel with BATCH=32, SEQ_LEN=128, N_CLASSES=2.
- Compute argmax over positions with (token != 0) mask
- Gather 2-class logits from (source_batch, last_pos)
- 2-class log-softmax with bf16 round-trip
- NLL loss with ignore-index masking
- Return: argmax, selected logits, count, mean loss, ne2 mask, one-hot eq mask
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _last_token_xent_kernel(
    logits_ptr,        # bf16 [rows, N_CLASSES]
    tokens_ptr,        # i64 [BATCH, SEQ_LEN]
    row_index_ptr,     # i64 [BATCH]
    labels_ptr,        # i64 [BATCH]
    ignored_value_ptr, # f32 scalar
    argmax_ptr,        # i64 [BATCH]
    selected_ptr,      # bf16 [BATCH, N_CLASSES]
    count_ptr,         # f32 scalar
    loss_ptr,          # f32 scalar
    ne2_ptr,           # b8 [BATCH]
    eq_ptr,            # b8 [BATCH, N_CLASSES]
    SEQ_LEN: ct.Constant[int],
    N_CLASSES: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    # Load token mask [BLOCK_B, BLOCK_S]
    tokens = ct.load(tokens_ptr, index=(0, 0), shape=(BLOCK_B, BLOCK_S))
    seq_1d = ct.arange(BLOCK_S, dtype=ct.int64)
    seq_2d = ct.reshape(seq_1d, (1, BLOCK_S))
    zero_i64 = ct.full((BLOCK_B, BLOCK_S), 0, dtype=ct.int64)
    positions = ct.where(tokens != 0, seq_2d, zero_i64)
    last_pos = ct.max(positions, axis=1)
    ct.store(argmax_ptr, index=(0,), tile=last_pos)

    # Load source batch indices (BLOCK_B,)
    source_batch = ct.load(row_index_ptr, index=(0,), shape=(BLOCK_B,))
    row = source_batch * SEQ_LEN + last_pos  # (BLOCK_B,)
    c_idx = ct.arange(N_CLASSES, dtype=ct.int64)
    row_2d = ct.reshape(row, (BLOCK_B, 1))
    c_2d = ct.reshape(c_idx, (1, N_CLASSES))
    # 2D gather from logits [rows, N_CLASSES] using (row, c) indices
    logit_bf16 = ct.gather(logits_ptr, (row_2d, c_2d))
    ct.store(selected_ptr, index=(0, 0), tile=logit_bf16)

    logit_f32 = ct.astype(logit_bf16, ct.float32)
    row_max = ct.max(logit_f32, axis=1, keepdims=True)
    sub = logit_f32 - row_max
    exp = ct.exp(sub)
    denom = ct.sum(exp, axis=1, keepdims=True)
    log_denom = ct.log(denom)
    logp = sub - log_denom
    logp_bf16 = ct.astype(logp, ct.bfloat16)
    logp_roundtrip = ct.astype(logp_bf16, ct.float32)

    labels = ct.load(labels_ptr, index=(0,), shape=(BLOCK_B,))
    ignore_val_tile = ct.full((BLOCK_B,), -100, dtype=ct.int64)
    valid = labels != ignore_val_tile
    zero_i64_1d = ct.full((BLOCK_B,), 0, dtype=ct.int64)
    safe_label = ct.where(valid, labels, zero_i64_1d)
    # For N_CLASSES=2, target = where(safe_label==0, logp[:,0], logp[:,1])
    # Do this by broadcasting safe_label over N_CLASSES and selecting.
    safe_label_2d_for_target = ct.reshape(safe_label, (BLOCK_B, 1))
    c_bcast = ct.reshape(c_idx, (1, N_CLASSES))
    is_target = safe_label_2d_for_target == c_bcast
    target_expanded = ct.where(is_target, logp_roundtrip, ct.full((BLOCK_B, N_CLASSES), 0.0, dtype=ct.float32))
    target = ct.sum(target_expanded, axis=1)
    neg = 0.0 - target
    # ignored is a 1-element 1D tensor (reshaped in Python). Load and broadcast.
    ignored_1elem = ct.load(ignored_value_ptr, index=(0,), shape=(1,))
    # Broadcast to (BLOCK_B,) via + with a zero tile
    ignored_bcast = ct.full((BLOCK_B,), 0.0, dtype=ct.float32) + ct.reshape(ignored_1elem, (1,))
    row_loss = ct.where(valid, neg, ignored_bcast)
    valid_f32 = ct.where(
        valid,
        ct.full((BLOCK_B,), 1.0, dtype=ct.float32),
        ct.full((BLOCK_B,), 0.0, dtype=ct.float32),
    )
    total_loss = ct.sum(row_loss)
    total_valid = ct.sum(valid_f32)
    # count_ptr and loss_ptr are 1-element 1D (reshaped in Python)
    ct.store(count_ptr, index=(0,), tile=ct.reshape(total_valid, (1,)))
    ct.store(loss_ptr, index=(0,), tile=ct.reshape(total_loss / total_valid, (1,)))

    ct.store(ne2_ptr, index=(0,), tile=valid)

    safe_label_2d = ct.reshape(safe_label, (BLOCK_B, 1))
    eq_result = safe_label_2d == c_2d
    ct.store(eq_ptr, index=(0, 0), tile=eq_result)


@oracle_impl(
    hardware="B200",
    point="6bcef7c8",
    BLOCK_B=32,
    BLOCK_S=128,
)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_S: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    view_shape = tuple(int(dim) for dim in shape0)
    batch = int(view_shape[0])
    seq_len = int(view_shape[1])
    n_classes = int(view_shape[2])
    mask_shape = tuple(int(dim) for dim in shape2)

    argmax = torch.empty((batch,), device=arg0_1.device, dtype=torch.int64)
    selected = torch.empty_strided(
        (batch, n_classes), (n_classes, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    count = torch.empty((), device=arg0_1.device, dtype=torch.float32)
    loss = torch.empty((), device=arg0_1.device, dtype=torch.float32)
    ne2 = torch.empty((batch,), device=arg0_1.device, dtype=torch.bool)
    eq = torch.empty_strided(
        mask_shape, (n_classes, 1),
        device=arg0_1.device, dtype=torch.bool,
    )

    # Reshape scalar tensors to 1D 1-element for cuTile
    ignored_1d = arg4_1.view(1)
    count_1d = count.view(1)
    loss_1d = loss.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _last_token_xent_kernel,
        (
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            ignored_1d,
            argmax,
            selected,
            count_1d,
            loss_1d,
            ne2,
            eq,
            seq_len,
            n_classes,
            BLOCK_B,
            BLOCK_S,
        ),
    )
    return argmax, selected, count, loss, ne2.view(batch, 1), eq
