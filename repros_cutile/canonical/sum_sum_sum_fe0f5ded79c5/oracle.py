"""cuTile port of sum_sum_sum_fe0f5ded79c5: GPT-Neo LN-backward + embed scatter.

Mirrors Triton's 3-kernel structure:
  1. `_init_outputs_kernel`: zero sum_3, sum_4, out_position, out_vocab.
  2. `_row_stats_kernel`: per-row row_sum (of weighted) and row_dot
     (weighted * normalized).
  3. `_seq_scatter_reduce_kernel`: recompute per-element algebra, atomic-add
     into (out_sum3, out_sum4) per hidden col, out_position (per position row)
     and out_vocab (per token row).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 2048
POSITION_ROWS = 2048
VOCAB_ROWS = 50257


@ct.kernel
def _init_outputs_kernel(
    out_sum3_ptr,        # f32 [HIDDEN]
    out_sum4_ptr,        # f32 [HIDDEN]
    out_position_ptr,    # f32 [POSITION_ROWS * HIDDEN]
    out_vocab_ptr,       # f32 [VOCAB_ROWS * HIDDEN]
    VOCAB_TOTAL: ct.Constant[int],
    POSITION_TOTAL: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    v_lim = ct.full(shape=(BLOCK,), fill_value=VOCAB_TOTAL, dtype=ct.int32)
    v_valid = offsets < v_lim
    zeros = ct.zeros((BLOCK,), dtype=ct.float32)
    ct.scatter(out_vocab_ptr, offsets, zeros, mask=v_valid)

    p_lim = ct.full(shape=(BLOCK,), fill_value=POSITION_TOTAL, dtype=ct.int32)
    p_valid = offsets < p_lim
    ct.scatter(out_position_ptr, offsets, zeros, mask=p_valid)

    h_lim = ct.full(shape=(BLOCK,), fill_value=HIDDEN_, dtype=ct.int32)
    h_valid = offsets < h_lim
    ct.scatter(out_sum3_ptr, offsets, zeros, mask=h_valid)
    ct.scatter(out_sum4_ptr, offsets, zeros, mask=h_valid)


@ct.kernel
def _row_stats_kernel(
    arg0_ptr,        # bf16 [rows, HIDDEN]
    arg1_ptr,        # bf16 [rows, HIDDEN]
    arg2_ptr,        # bf16 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    saved_a_ptr,     # f32 [rows, HIDDEN]
    saved_b_ptr,     # f32 [SEQ, HIDDEN]
    mean_ptr,        # f32 [rows]
    invstd_ptr,      # f32 [rows]
    row_sum_ptr,     # f32 [rows]
    row_dot_ptr,     # f32 [rows]
    HIDDEN_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    seq = row - (row // SEQ_) * SEQ_

    a0 = ct.astype(ct.load(arg0_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    a1 = ct.astype(ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    a2 = ct.astype(ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    source = a0 + a1 + a2
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    weighted = source * weight_2d

    sa = ct.load(saved_a_ptr, index=(row, 0), shape=(1, BLOCK_H))
    sb_pre = ct.load(saved_b_ptr, index=(seq, 0), shape=(1, BLOCK_H))
    norm_pre = sa + sb_pre
    mean_v = ct.load(mean_ptr, index=(row,), shape=(1,))
    mean_2d = ct.reshape(mean_v, (1, 1))
    invstd_v = ct.load(invstd_ptr, index=(row,), shape=(1,))
    invstd_2d = ct.reshape(invstd_v, (1, 1))
    normalized = (norm_pre - mean_2d) * invstd_2d

    row_sum = ct.sum(weighted, axis=1, keepdims=False)
    row_dot = ct.sum(weighted * normalized, axis=1, keepdims=False)
    ct.store(row_sum_ptr, index=(row,), tile=ct.reshape(row_sum, (1,)))
    ct.store(row_dot_ptr, index=(row,), tile=ct.reshape(row_dot, (1,)))


@ct.kernel
def _seq_scatter_reduce_kernel(
    arg0_ptr,           # bf16 [rows, HIDDEN]
    arg1_ptr,           # bf16 [rows, HIDDEN]
    arg2_ptr,           # bf16 [rows, HIDDEN]
    weight_ptr,         # f32 [HIDDEN]
    saved_a_ptr,        # f32 [rows, HIDDEN]
    saved_b_ptr,        # f32 [SEQ, HIDDEN]
    mean_ptr,           # f32 [rows]
    invstd_ptr,         # f32 [rows]
    upstream_ptr,       # f32 [rows, HIDDEN]
    position_index_ptr, # i64 [SEQ]
    token_index_ptr,    # i64 [rows]
    row_sum_ptr,        # f32 [rows]
    row_dot_ptr,        # f32 [rows]
    out_sum3_ptr,       # f32 [HIDDEN]
    out_sum4_ptr,       # f32 [HIDDEN]
    out_position_ptr,   # f32 [POSITION_ROWS * HIDDEN]
    out_vocab_ptr,      # f32 [VOCAB_ROWS * HIDDEN]
    HIDDEN_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    BATCH_: ct.Constant[int],
    VOCAB_ROWS_: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    pid_c = ct.bid(0)
    seq = ct.bid(1)

    batches = ct.arange(BLOCK_B, dtype=ct.int32)
    cols = pid_c * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    batch_lim = ct.full(shape=(BLOCK_B,), fill_value=BATCH_, dtype=ct.int32)
    col_lim = ct.full(shape=(BLOCK_C,), fill_value=HIDDEN_, dtype=ct.int32)
    row_active = batches < batch_lim
    col_active = cols < col_lim
    row_active_2d = ct.reshape(row_active, (BLOCK_B, 1))
    col_active_2d = ct.reshape(col_active, (1, BLOCK_C))
    active = row_active_2d & col_active_2d

    rows_full = batches * SEQ_ + seq  # shape (BLOCK_B,)
    rows_bc = ct.broadcast_to(ct.reshape(rows_full, (BLOCK_B, 1)),
                               (BLOCK_B, BLOCK_C))
    cols_bc = ct.broadcast_to(ct.reshape(cols, (1, BLOCK_C)),
                               (BLOCK_B, BLOCK_C))

    a0 = ct.astype(
        ct.gather(arg0_ptr, (rows_bc, cols_bc), mask=active), ct.float32,
    )
    a1 = ct.astype(
        ct.gather(arg1_ptr, (rows_bc, cols_bc), mask=active), ct.float32,
    )
    a2 = ct.astype(
        ct.gather(arg2_ptr, (rows_bc, cols_bc), mask=active), ct.float32,
    )
    source = a0 + a1 + a2

    weight = ct.load(weight_ptr, index=(pid_c,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    weighted = source * weight_2d

    sa = ct.gather(saved_a_ptr, (rows_bc, cols_bc), mask=active)
    seq_bc = ct.full(shape=(BLOCK_B, BLOCK_C), fill_value=seq, dtype=ct.int32)
    sb = ct.gather(saved_b_ptr, (seq_bc, cols_bc), mask=active)
    mean_v = ct.load(mean_ptr, index=(0,), shape=(BLOCK_B,),
                      padding_mode=ct.PaddingMode.ZERO)
    invstd_v = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_B,),
                        padding_mode=ct.PaddingMode.ZERO)
    # Only rows for this seq: rows_full = batches * SEQ + seq
    mean_batches = ct.gather(mean_ptr, rows_full)  # (BLOCK_B,)
    invstd_batches = ct.gather(invstd_ptr, rows_full)
    mean_2d = ct.reshape(mean_batches, (BLOCK_B, 1))
    invstd_2d = ct.reshape(invstd_batches, (BLOCK_B, 1))
    normalized = (sa + sb - mean_2d) * invstd_2d

    row_sum = ct.gather(row_sum_ptr, rows_full)
    row_dot = ct.gather(row_dot_ptr, rows_full)
    rs_2d = ct.reshape(row_sum, (BLOCK_B, 1))
    rd_2d = ct.reshape(row_dot, (BLOCK_B, 1))

    mul_scaled = weighted * HIDDEN_
    sub1 = mul_scaled - rs_2d
    mul4 = normalized * rd_2d
    sub2 = sub1 - mul4
    div = invstd_2d / HIDDEN_
    producer = div * sub2
    upstream = ct.gather(upstream_ptr, (rows_bc, cols_bc), mask=active)
    add3 = upstream + producer

    # Column atomic adds: sum3, sum4.
    # OOB contributions are zeroed via mask before reduce; the atomic_add is
    # unmasked (invalid columns get 0 which is a no-op).
    zeros_2d = ct.zeros((BLOCK_B, BLOCK_C), dtype=ct.float32)
    sn_masked = ct.where(active, source * normalized, zeros_2d)
    s_masked = ct.where(active, source, zeros_2d)
    col_sn = ct.sum(sn_masked, axis=0, keepdims=False)
    col_s = ct.sum(s_masked, axis=0, keepdims=False)
    # Send OOB cols to index 0 with value 0 (no-op) via redirect.
    zeros_c = ct.zeros((BLOCK_C,), dtype=ct.float32)
    safe_cols = ct.where(col_active, cols,
                          ct.zeros((BLOCK_C,), dtype=ct.int32))
    col_sn_safe = ct.where(col_active, col_sn, zeros_c)
    col_s_safe = ct.where(col_active, col_s, zeros_c)
    ct.atomic_add(out_sum3_ptr, (safe_cols,), col_sn_safe)
    ct.atomic_add(out_sum4_ptr, (safe_cols,), col_s_safe)

    # Position scatter.
    position_index_scalar = ct.load(position_index_ptr, index=(seq,), shape=(1,))
    # Broadcast to 1D of BLOCK_C.
    pos_idx_bc = ct.broadcast_to(
        ct.astype(ct.reshape(position_index_scalar, (1,)), ct.int32),
        (BLOCK_C,),
    )
    pos_off = pos_idx_bc * HIDDEN_ + cols
    add3_masked = ct.where(active, add3, zeros_2d)
    add3_col_sum = ct.sum(add3_masked, axis=0, keepdims=False)
    add3_col_sum_safe = ct.where(col_active, add3_col_sum, zeros_c)
    pos_off_safe = ct.where(col_active, pos_off,
                              ct.zeros((BLOCK_C,), dtype=ct.int32))
    ct.atomic_add(out_position_ptr, (pos_off_safe,), add3_col_sum_safe)

    # Token scatter into vocab.
    token_batches = ct.gather(token_index_ptr, rows_full)  # (BLOCK_B,) i64
    token_i32 = ct.astype(token_batches, ct.int32)
    token_bc = ct.broadcast_to(ct.reshape(token_i32, (BLOCK_B, 1)),
                                 (BLOCK_B, BLOCK_C))
    zero_i32 = ct.full(shape=(BLOCK_B, BLOCK_C), fill_value=0, dtype=ct.int32)
    vocab_lim_i32 = ct.full(shape=(BLOCK_B, BLOCK_C), fill_value=VOCAB_ROWS_, dtype=ct.int32)
    tok_valid = (token_bc >= zero_i32) & (token_bc < vocab_lim_i32)
    scatter_mask = active & tok_valid
    vocab_off = token_bc * HIDDEN_ + cols_bc
    vocab_off_safe = ct.where(scatter_mask, vocab_off, zero_i32)
    add3_safe = ct.where(scatter_mask, add3, zeros_2d)
    ct.atomic_add(out_vocab_ptr, (vocab_off_safe,), add3_safe)


def _shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="cb38dfd8")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     arg9_1, arg10_1,
     shape0, shape1, shape2, shape3, shape4, shape5) = inputs
    device = arg0_1.device
    hidden = HIDDEN
    n = BATCH
    seq = SEQ
    rows = n * seq

    # Flatten [rows, hidden] views (contiguous per repro spec).
    a0_2d = arg0_1.view(rows, hidden)
    a1_2d = arg1_1.view(rows, hidden)
    a2_2d = arg2_1.view(rows, hidden)
    sa_2d = arg4_1.view(rows, hidden)
    sb_2d = arg5_1.view(seq, hidden)
    mean_flat = arg6_1.view(rows)
    invstd_flat = arg7_1.view(rows)
    upstream_2d = arg8_1.view(rows, hidden)
    pos_idx_flat = arg9_1.view(seq)
    tok_idx_flat = arg10_1.view(rows)

    row_sum = torch.empty((rows,), device=device, dtype=torch.float32)
    row_dot = torch.empty((rows,), device=device, dtype=torch.float32)
    out_sum3 = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS * hidden,), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS * hidden,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    INIT_BLOCK = 1024
    vocab_total = VOCAB_ROWS * hidden
    position_total = POSITION_ROWS * hidden
    ct.launch(
        stream, ((vocab_total + INIT_BLOCK - 1) // INIT_BLOCK, 1, 1),
        _init_outputs_kernel,
        (out_sum3, out_sum4, out_position, out_vocab,
         vocab_total, position_total, hidden, INIT_BLOCK),
    )

    BLOCK_H = 2048
    ct.launch(
        stream, (rows, 1, 1), _row_stats_kernel,
        (a0_2d, a1_2d, a2_2d, arg3_1, sa_2d, sb_2d,
         mean_flat, invstd_flat, row_sum, row_dot,
         hidden, seq, BLOCK_H),
    )

    BLOCK_B = 32
    BLOCK_C = 32
    ct.launch(
        stream, ((hidden + BLOCK_C - 1) // BLOCK_C, seq, 1),
        _seq_scatter_reduce_kernel,
        (a0_2d, a1_2d, a2_2d, arg3_1, sa_2d, sb_2d,
         mean_flat, invstd_flat, upstream_2d, pos_idx_flat, tok_idx_flat,
         row_sum, row_dot, out_sum3, out_sum4, out_position, out_vocab,
         hidden, seq, n, VOCAB_ROWS, BLOCK_B, BLOCK_C),
    )

    return (out_sum3, out_sum4,
            out_position.view(POSITION_ROWS, hidden),
            out_vocab.view(VOCAB_ROWS, hidden))
