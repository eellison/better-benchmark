"""cuTile port of amax_sum_sum_52903d400fef: shifted-label CE loss over
Roberta bf16 logits.

Three sequential kernels:
1. `_pad_right_kernel` — pad labels to `(batch, seq_len+1)` with -100.
2. `_shifted_xent_stats_rows_kernel` — per-row amax + log-sum-exp + target
   lookup over the vocab dim, streaming across the vocab (50265 cols) in
   BLOCK_N chunks.
3. `_mean_reduce_kernel` — total loss / valid-count -> scalar mean, count.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ_LEN = 512
N_ROWS = BATCH * SEQ_LEN
N_COLS = 50265


@ct.kernel
def _pad_right_kernel(
    labels_ptr,
    padded_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    # padded_ptr is (BATCH * (SEQ_LEN + 1),) — length = BATCH*513 = 16416.
    # labels_ptr is (BATCH * SEQ_LEN,) = 16384. Pad each row's last element.
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    total_padded = BATCH * (SEQ_LEN + 1)
    valid = offsets < total_padded
    row = offsets // (SEQ_LEN + 1)
    col = offsets - row * (SEQ_LEN + 1)
    is_col = col < SEQ_LEN
    # Compute source index for valid entries; for OOB use 0
    src = row * SEQ_LEN + col
    zero_i64 = ct.full((BLOCK,), 0, dtype=ct.int64)
    src_i64 = ct.astype(src, ct.int64)
    safe_src = ct.where(is_col & valid, src_i64, zero_i64)
    values = ct.gather(labels_ptr, safe_src, mask=is_col & valid, padding_value=-100)
    minus100 = ct.full((BLOCK,), -100, dtype=ct.int64)
    values = ct.where(is_col, values, minus100)
    # Scatter into padded_ptr; keep out-of-bounds by mask
    ct.scatter(padded_ptr, offsets, values, mask=valid)


@ct.kernel
def _shifted_xent_stats_rows_kernel(
    labels_ptr,      # i64 (BATCH*SEQ_LEN,)
    logits_ptr,      # bf16 (N_ROWS, LOGITS_STRIDE)
    amax_ptr,        # f32 (N_ROWS,)
    log_ptr,         # f32 (N_ROWS,)
    loss_ptr,        # f32 (N_ROWS,)
    valid_ptr,       # f32 (N_ROWS,)
    N_COLS_: ct.Constant[int],
    LOGITS_STRIDE: ct.Constant[int],
    SEQ_LEN_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    pos = row % SEQ_LEN_
    has_next = pos < (SEQ_LEN_ - 1)

    # Load "next" label; guard OOB when row+1 is at last position
    next_row_i64 = ct.full((1,), row + 1, dtype=ct.int64)
    zero_i64 = ct.full((1,), 0, dtype=ct.int64)
    label_gather_idx = ct.where(
        ct.full((1,), has_next, dtype=ct.bool_), next_row_i64, zero_i64
    )
    label = ct.gather(labels_ptr, label_gather_idx, mask=ct.full((1,), has_next, dtype=ct.bool_), padding_value=-100)
    minus100 = ct.full((1,), -100, dtype=ct.int64)
    is_valid_v = label != minus100
    safe_label = ct.where(is_valid_v, label, ct.full((1,), 0, dtype=ct.int64))
    safe_label_i32 = ct.astype(safe_label, ct.int32)

    # Load target logit at [row, safe_label]
    row_i32 = ct.full((1,), row, dtype=ct.int32)
    target_bf = ct.gather(logits_ptr, (row_i32, safe_label_i32),
                          mask=is_valid_v, padding_value=0)
    target_f = ct.astype(target_bf, ct.float32)

    # Streaming online softmax stats over the vocab dim
    neg_inf = ct.full((1,), float("-inf"), dtype=ct.float32)
    row_max = ct.reshape(neg_inf, (1,))
    denom = ct.zeros((1,), dtype=ct.float32)

    num_blocks = (N_COLS_ + BLOCK_N - 1) // BLOCK_N
    for i in range(num_blocks):
        cols = ct.arange(BLOCK_N, dtype=ct.int32) + i * BLOCK_N
        col_mask = cols < N_COLS_
        # Load logits[row, cols]: use ct.gather over (row, col).
        row_bcast = ct.full((BLOCK_N,), row, dtype=ct.int32)
        logits_bf = ct.gather(logits_ptr, (row_bcast, cols),
                              mask=col_mask, padding_value=0)
        logits = ct.astype(logits_bf, ct.float32)
        neg_inf_tile = ct.full((BLOCK_N,), float("-inf"), dtype=ct.float32)
        logits = ct.where(col_mask, logits, neg_inf_tile)
        block_max_v = ct.max(logits)  # scalar
        block_max = ct.reshape(block_max_v, (1,))
        new_max = ct.maximum(row_max, block_max)
        denom = denom * ct.exp(row_max - new_max)
        # Sum exp(logits - new_max) over the block
        new_max_bcast = ct.full((BLOCK_N,), 0.0, dtype=ct.float32) + ct.reshape(new_max, (1,))
        denom_add = ct.sum(ct.exp(logits - new_max_bcast))
        denom = denom + ct.reshape(denom_add, (1,))
        row_max = new_max

    log_denom = ct.log(denom)
    loss = row_max + log_denom - ct.reshape(target_f, (1,))
    zero_f = ct.zeros((1,), dtype=ct.float32)
    one_f = ct.full((1,), 1.0, dtype=ct.float32)
    loss = ct.where(is_valid_v, loss, zero_f)
    valid_out = ct.where(is_valid_v, one_f, zero_f)

    ct.store(amax_ptr, index=(row,), tile=row_max)
    ct.store(log_ptr, index=(row,), tile=log_denom)
    ct.store(loss_ptr, index=(row,), tile=loss)
    ct.store(valid_ptr, index=(row,), tile=valid_out)


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    count_out_ptr,
    div_out_ptr,
    N_ROWS_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                     padding_mode=ct.PaddingMode.ZERO)
    valid = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,),
                    padding_mode=ct.PaddingMode.ZERO)
    total_loss = ct.sum(losses)
    total_valid = ct.sum(valid)
    ct.store(count_out_ptr, index=(0,), tile=ct.reshape(total_valid, (1,)))
    ct.store(div_out_ptr, index=(0,),
             tile=ct.reshape(total_loss / total_valid, (1,)))


# 68a9ca47: RobertaForCausalLM train bf16 logits [16384,50272] -> [:, :50265].
@oracle_impl(hardware="B200", point="68a9ca47", BLOCK_N=4096)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = tuple(int(d) for d in shape_3d)
    n_rows = int(labels.numel())
    seq_len = int(labels.shape[1])
    batch = int(labels.shape[0])
    n_cols = int(shape_2d[1])

    logits_view = logits[:, :n_cols].view(view_shape)
    padded_labels = torch.empty(
        (batch, seq_len + 1), device=labels.device, dtype=torch.int64,
    )
    amax = torch.empty(
        (n_rows, 1), device=logits.device, dtype=torch.float32,
    )
    log = torch.empty(
        (n_rows, 1), device=logits.device, dtype=torch.float32,
    )
    loss_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    count = torch.empty((), device=logits.device, dtype=torch.float32)
    div = torch.empty((), device=logits.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    pad_block = 512
    total_padded = batch * (seq_len + 1)
    ct.launch(
        stream,
        ((total_padded + pad_block - 1) // pad_block, 1, 1),
        _pad_right_kernel,
        (labels.reshape(-1), padded_labels.reshape(-1), pad_block),
    )
    # For gather-friendly access, the logits argument to cuTile should
    # keep its native strided [16384, 50272] shape.
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _shifted_xent_stats_rows_kernel,
        (labels.reshape(-1), logits, amax.reshape(-1), log.reshape(-1),
         loss_per_row, valid_per_row,
         n_cols, int(logits.stride(0)), seq_len, BLOCK_N),
    )
    # Round up n_rows to next power of two for the reduce
    m = 1
    while m < n_rows:
        m *= 2
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, count.reshape(1), div.reshape(1),
         n_rows, m),
    )
    return logits_view, padded_labels, amax, log, count, div
