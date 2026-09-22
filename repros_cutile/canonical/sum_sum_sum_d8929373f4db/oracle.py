"""cuTile port of sum_sum_sum_d8929373f4db: DistilBERT LN-backward + masked embedding scatter.

Structure:
  * Precompute base copy: out_vocab starts as arg0_1.f32()
  * Precompute row_sum, row_dot per (batch, seq) row via row-wise kernel
  * Scatter kernel: iterate over (seq, batch_block, hidden_col_block):
    - Compute mul1 = source * dropout, mul2 = mul1 * weight
    - Compute mul4 = (saved + position_saved - mean) * invstd
    - Compute producer = (invstd/H) * (mul2*H - row_sum - mul4*row_dot)
    - Atomic add row-sums of (mul1*mul4) into sum_3
    - Atomic add row-sums of (mul1) into sum_4
    - Atomic add sum-over-batch of producer into position table at pos_index[seq]
    - Atomic add producer values into vocab table at token_index[b, seq]
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
SEQ = 128
HIDDEN = 768
ROWS = BATCH * SEQ  # 32768
POSITION_ROWS = 512
VOCAB_ROWS = 30522
DROPOUT_SCALE = 1.1111111111111112
BLOCK_H = 1024  # pow2 >= 768


@ct.kernel
def _row_stats_kernel(
    arg1_ptr, arg2_ptr, arg3_ptr, arg4_ptr,  # [ROWS, HIDDEN]
    keep_ptr,      # b8 [ROWS, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    saved_ptr,     # f32 [ROWS, HIDDEN]
    position_saved_ptr,  # f32 [SEQ, HIDDEN]
    mean_ptr,      # f32 [ROWS]
    invstd_ptr,    # f32 [ROWS]
    row_sum_ptr,   # f32 [ROWS]
    row_dot_ptr,   # f32 [ROWS]
    HIDDEN_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    h = ct.arange(BLOCK_H_, dtype=ct.int32)
    active = h < HIDDEN_
    seq = row - (row // SEQ_) * SEQ_
    zero_f = ct.full((BLOCK_H_,), 0.0, dtype=ct.float32)

    a1 = ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    a2 = ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    a3 = ct.load(arg3_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    a4 = ct.load(arg4_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    source = (ct.astype(a2, ct.float32) + ct.astype(a1, ct.float32)
              + ct.astype(a3, ct.float32) + ct.astype(a4, ct.float32))
    source_1d = ct.reshape(source, (BLOCK_H_,))

    keep = ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                   padding_mode=ct.PaddingMode.ZERO)
    keep_f = ct.astype(keep, ct.float32)
    keep_1d = ct.reshape(keep_f, (BLOCK_H_,))
    dropout = keep_1d * DROPOUT_SCALE_
    mul1 = source_1d * dropout

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,),
                     padding_mode=ct.PaddingMode.ZERO)
    mul2 = mul1 * weight

    saved = ct.load(saved_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                    padding_mode=ct.PaddingMode.ZERO)
    saved_1d = ct.reshape(saved, (BLOCK_H_,))
    pos_saved = ct.load(position_saved_ptr, index=(seq, 0), shape=(1, BLOCK_H_),
                        padding_mode=ct.PaddingMode.ZERO)
    pos_saved_1d = ct.reshape(pos_saved, (BLOCK_H_,))

    mean = ct.load(mean_ptr, index=(row,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))
    mean_s = ct.reshape(mean, (1,))
    invstd_s = ct.reshape(invstd, (1,))

    add3 = saved_1d + pos_saved_1d
    sub = add3 - mean_s
    mul4 = sub * invstd_s

    mul2_masked = ct.where(active, mul2, zero_f)
    mul2_mul4_masked = ct.where(active, mul2 * mul4, zero_f)
    row_sum = ct.sum(mul2_masked)
    row_dot = ct.sum(mul2_mul4_masked)
    ct.store(row_sum_ptr, index=(row,), tile=ct.reshape(row_sum, (1,)))
    ct.store(row_dot_ptr, index=(row,), tile=ct.reshape(row_dot, (1,)))


@ct.kernel
def _scatter_kernel(
    arg1_ptr, arg2_ptr, arg3_ptr, arg4_ptr,  # bf16 [ROWS, HIDDEN]
    keep_ptr,       # b8 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    saved_ptr,      # f32 [ROWS, HIDDEN]
    position_saved_ptr,  # f32 [SEQ, HIDDEN]
    mean_ptr,       # f32 [ROWS]
    invstd_ptr,     # f32 [ROWS]
    position_index_ptr,  # i64 [SEQ]
    token_index_ptr,     # i64 [ROWS]
    row_sum_ptr,    # f32 [ROWS]
    row_dot_ptr,    # f32 [ROWS]
    out_sum3_ptr,   # f32 [HIDDEN]
    out_sum4_ptr,   # f32 [HIDDEN]
    out_position_ptr,  # f32 [POSITION_ROWS, HIDDEN]
    out_vocab_ptr,     # f32 [VOCAB_ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    BATCH_: ct.Constant[int],
    VOCAB_ROWS_: ct.Constant[int],
    POSITION_ROWS_: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
    HIDDEN_F: ct.Constant[float],
    BLOCK_H_: ct.Constant[int],
):
    b = ct.bid(0)  # batch idx
    seq = ct.bid(1)
    row = b * SEQ_ + seq

    h = ct.arange(BLOCK_H_, dtype=ct.int32)
    active = h < HIDDEN_
    zero_f = ct.full((BLOCK_H_,), 0.0, dtype=ct.float32)

    a1 = ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    a2 = ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    a3 = ct.load(arg3_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    a4 = ct.load(arg4_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    source = (ct.astype(a2, ct.float32) + ct.astype(a1, ct.float32)
              + ct.astype(a3, ct.float32) + ct.astype(a4, ct.float32))
    source_1d = ct.reshape(source, (BLOCK_H_,))

    keep = ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                   padding_mode=ct.PaddingMode.ZERO)
    keep_f = ct.astype(keep, ct.float32)
    keep_1d = ct.reshape(keep_f, (BLOCK_H_,))
    dropout = keep_1d * DROPOUT_SCALE_
    mul1 = source_1d * dropout

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,),
                     padding_mode=ct.PaddingMode.ZERO)
    mul2 = mul1 * weight

    saved = ct.load(saved_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                    padding_mode=ct.PaddingMode.ZERO)
    saved_1d = ct.reshape(saved, (BLOCK_H_,))
    pos_saved = ct.load(position_saved_ptr, index=(seq, 0), shape=(1, BLOCK_H_),
                        padding_mode=ct.PaddingMode.ZERO)
    pos_saved_1d = ct.reshape(pos_saved, (BLOCK_H_,))

    mean = ct.load(mean_ptr, index=(row,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))
    mean_s = ct.reshape(mean, (1,))
    invstd_s = ct.reshape(invstd, (1,))

    add3 = saved_1d + pos_saved_1d
    sub = add3 - mean_s
    mul4 = sub * invstd_s  # (BLOCK_H,)

    row_sum = ct.load(row_sum_ptr, index=(row,), shape=(1,))
    row_dot = ct.load(row_dot_ptr, index=(row,), shape=(1,))
    row_sum_s = ct.reshape(row_sum, (1,))
    row_dot_s = ct.reshape(row_dot, (1,))

    mul3 = mul2 * HIDDEN_F
    sub1 = mul3 - row_sum_s
    mul6 = mul4 * row_dot_s
    sub2 = sub1 - mul6
    div = invstd_s / HIDDEN_F
    producer = ct.where(active, div * sub2, zero_f)

    # sum_3, sum_4: atomic add mul1*mul4 and mul1 per-column into out_sum3/4
    mul1_mul4 = ct.where(active, mul1 * mul4, zero_f)
    mul1_masked = ct.where(active, mul1, zero_f)
    ct.atomic_add(out_sum3_ptr, (h,), mul1_mul4)
    ct.atomic_add(out_sum4_ptr, (h,), mul1_masked)

    # Position scatter: only accumulate for seq=0 (once per seq).
    # But out_position_ptr is written from sum-over-batch of producer per seq.
    # So we scatter per (b, seq); result is naturally sum-over-batch at pos index.
    position_idx = ct.load(position_index_ptr, index=(seq,), shape=(1,))
    position_idx_i32 = ct.astype(position_idx, ct.int32)
    pos_scalar = ct.reshape(position_idx_i32, (1,))
    # Position validity: 0 <= idx < POSITION_ROWS and idx != -1 (i.e. >=0)
    valid_pos = (pos_scalar >= ct.full((1,), 0, dtype=ct.int32)) & \
                (pos_scalar < ct.full((1,), POSITION_ROWS_, dtype=ct.int32))
    # Redirect invalid to out-of-bounds
    safe_pos = ct.where(valid_pos, pos_scalar,
                        ct.full((1,), POSITION_ROWS_, dtype=ct.int32))
    # Broadcast safe_pos across BLOCK_H so we can index [row, col]
    safe_pos_b = ct.broadcast_to(safe_pos, (BLOCK_H_,))
    ct.atomic_add(out_position_ptr, (safe_pos_b, h), producer)

    # Vocab scatter
    token_idx = ct.load(token_index_ptr, index=(row,), shape=(1,))
    token_idx_i32 = ct.astype(token_idx, ct.int32)
    tok_scalar = ct.reshape(token_idx_i32, (1,))
    valid_tok = (tok_scalar >= ct.full((1,), 0, dtype=ct.int32)) & \
                (tok_scalar < ct.full((1,), VOCAB_ROWS_, dtype=ct.int32)) & \
                (tok_scalar != ct.full((1,), 0, dtype=ct.int32))
    safe_tok = ct.where(valid_tok, tok_scalar,
                        ct.full((1,), VOCAB_ROWS_, dtype=ct.int32))
    safe_tok_b = ct.broadcast_to(safe_tok, (BLOCK_H_,))
    ct.atomic_add(out_vocab_ptr, (safe_tok_b, h), producer)


@oracle_impl(hardware="B200", point="b2f45dd0")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
        arg9_1, arg10_1, arg11_1, arg12_1, *_shape_params,
    ) = inputs

    device = arg0_1.device

    # Precompute base copy: out_vocab starts as arg0_1.f32()
    out_vocab = arg0_1.to(torch.float32)
    out_position = torch.zeros((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_sum3 = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_sum4 = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)

    # Views for kernel
    a1 = arg1_1.view(ROWS, HIDDEN)
    a2 = arg2_1.view(ROWS, HIDDEN)
    a3 = arg3_1.view(ROWS, HIDDEN)
    a4 = arg4_1.view(ROWS, HIDDEN)
    keep = arg5_1.view(ROWS, HIDDEN)
    weight = arg6_1  # [HIDDEN]
    saved = arg7_1.view(ROWS, HIDDEN)
    position_saved = arg8_1.view(SEQ, HIDDEN)
    mean = arg9_1.view(ROWS)
    invstd = arg10_1.view(ROWS)
    position_index = arg11_1.view(POSITION_ROWS)  # [1, 512] -> flat 512
    token_index = arg12_1.view(ROWS)  # [256, 128]

    # Position slice: repro takes slice_1 = arg11_1[:, 0:128]; only first 128
    # position entries are used. But cuTile indexes at seq (0..127), so we
    # need position_index sliced to first 128. However arg11_1 is [1, 512],
    # so we need position_index[0:128].
    position_index_slice = arg11_1[0, :SEQ].contiguous()  # [128]

    stream = torch.cuda.current_stream()

    # Row stats first
    ct.launch(
        stream, (ROWS, 1, 1), _row_stats_kernel,
        (a1, a2, a3, a4, keep, weight, saved, position_saved,
         mean, invstd, row_sum, row_dot,
         HIDDEN, SEQ, DROPOUT_SCALE, BLOCK_H),
    )

    # Scatter kernel — grid (BATCH, SEQ, 1). We iterate over batch and seq.
    ct.launch(
        stream, (BATCH, SEQ, 1), _scatter_kernel,
        (a1, a2, a3, a4, keep, weight, saved, position_saved,
         mean, invstd, position_index_slice, token_index,
         row_sum, row_dot, out_sum3, out_sum4, out_position, out_vocab,
         HIDDEN, SEQ, BATCH, VOCAB_ROWS, POSITION_ROWS, DROPOUT_SCALE,
         float(HIDDEN), BLOCK_H),
    )

    return out_sum3, out_sum4, out_position, out_vocab
