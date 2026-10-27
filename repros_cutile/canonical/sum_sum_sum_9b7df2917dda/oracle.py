"""cuTile port of sum_sum_sum_9b7df2917dda: Roberta LN-backward with
embedding-gradient scatter.

Per row, compute the producer expression and atomic-add to position, segment,
and vocab tables. Uses `check_bounds=True` on `ct.atomic_add` (default) to
turn OOB indices into no-ops, encoding "inactive" as OOB.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ           # 16384
HIDDEN = 768                 # NOT a power of 2 — pad to 1024
BLOCK_H_PAD = 1024
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 50265
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _row_scatter_reduce_kernel(
    arg1_ptr, arg2_ptr, arg3_ptr, arg4_ptr,   # bf16 (ROWS, HIDDEN) each
    keep_ptr,        # b8 (ROWS, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    saved_ptr,       # f32 (ROWS, HIDDEN)
    scale_ptr,       # f32 (ROWS,)
    position_index_ptr,  # i64 (ROWS,)
    segment_index_ptr,   # i64 (ROWS,)
    vocab_mask_ptr,      # b8 (ROWS,)
    vocab_index_ptr,     # i64 (ROWS,)
    out_sum3_ptr,   # f32 (HIDDEN,)
    out_sum4_ptr,   # f32 (HIDDEN,)
    out_position_ptr,   # f32 (POSITION_ROWS, HIDDEN)
    out_segment_ptr,    # f32 (SEGMENT_ROWS, HIDDEN)
    out_vocab_ptr,      # f32 (VOCAB_ROWS, HIDDEN)
    HIDDEN_: ct.Constant[int],
    POSITION_ROWS_: ct.Constant[int],
    SEGMENT_ROWS_: ct.Constant[int],
    VOCAB_ROWS_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN_
    zero_col = ct.zeros((BLOCK_H,), dtype=ct.int32)

    arg1 = ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    arg2 = ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    arg3 = ct.load(arg3_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    arg4 = ct.load(arg4_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    keep = ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    saved = ct.load(saved_ptr, index=(row, 0), shape=(1, BLOCK_H),
                    padding_mode=ct.PaddingMode.ZERO)

    added = (
        ct.astype(arg2, ct.float32)
        + ct.astype(arg1, ct.float32)
        + ct.astype(arg3, ct.float32)
        + ct.astype(arg4, ct.float32)
    )
    added_1d = ct.reshape(added, (BLOCK_H,))
    keep_1d = ct.reshape(ct.astype(keep, ct.float32), (BLOCK_H,))
    mul1 = added_1d * (keep_1d * DROPOUT_SCALE)

    saved_1d = ct.reshape(saved, (BLOCK_H,))
    mul2 = mul1 * weight

    col_mask_f = ct.astype(col_mask, ct.float32)
    row_sum = ct.sum(mul2 * col_mask_f)
    row_dot = ct.sum(mul2 * saved_1d * col_mask_f)

    scale = ct.load(scale_ptr, index=(row,), shape=(1,))
    scale_scalar = ct.reshape(scale, (1,))
    scale_bcast = ct.full((BLOCK_H,), 0.0, dtype=ct.float32) + scale_scalar

    producer = scale_bcast * ((mul2 * HIDDEN_) - row_sum - (saved_1d * row_dot))
    # Zero out OOB columns of producer so they contribute 0 to reductions.
    zero_prod = ct.full((BLOCK_H,), 0.0, dtype=ct.float32)
    producer_masked = ct.where(col_mask, producer, zero_prod)

    # Sum reductions: bounds-checked atomic-add at col index (OOB skipped).
    ct.atomic_add(out_sum3_ptr, (cols,), ct.where(col_mask, mul1 * saved_1d, zero_prod))
    ct.atomic_add(out_sum4_ptr, (cols,), ct.where(col_mask, mul1, zero_prod))

    # Position scatter
    position_index_scalar = ct.load(position_index_ptr, index=(row,), shape=(1,))
    pos_idx_scalar = ct.reshape(position_index_scalar, (1,))
    zero_i = ct.full((1,), 0, dtype=ct.int64)
    pos_rows_i = ct.full((1,), POSITION_ROWS_, dtype=ct.int64)
    pos_active = (pos_idx_scalar >= zero_i) & (pos_idx_scalar < pos_rows_i) & (pos_idx_scalar != zero_i)
    # Encode inactive as OOB (POSITION_ROWS).
    safe_pos = ct.where(pos_active, pos_idx_scalar, pos_rows_i)
    pos_row_bcast = ct.astype(safe_pos, ct.int32) + zero_col
    ct.atomic_add(out_position_ptr, (pos_row_bcast, cols), producer_masked)

    # Segment scatter
    segment_index_scalar = ct.load(segment_index_ptr, index=(row,), shape=(1,))
    seg_idx_scalar = ct.reshape(segment_index_scalar, (1,))
    seg_rows_i = ct.full((1,), SEGMENT_ROWS_, dtype=ct.int64)
    minus_one_i = ct.full((1,), -1, dtype=ct.int64)
    seg_active = (seg_idx_scalar >= zero_i) & (seg_idx_scalar < seg_rows_i) & (seg_idx_scalar != minus_one_i)
    safe_seg = ct.where(seg_active, seg_idx_scalar, seg_rows_i)
    seg_row_bcast = ct.astype(safe_seg, ct.int32) + zero_col
    ct.atomic_add(out_segment_ptr, (seg_row_bcast, cols), producer_masked)

    # Vocab scatter
    vocab_mask_scalar = ct.load(vocab_mask_ptr, index=(row,), shape=(1,))
    vocab_idx_scalar_raw = ct.load(vocab_index_ptr, index=(row,), shape=(1,))
    vocab_idx_scalar = ct.reshape(vocab_idx_scalar_raw, (1,))
    vocab_mask_1 = ct.reshape(vocab_mask_scalar, (1,))
    vocab_rows_i = ct.full((1,), VOCAB_ROWS_, dtype=ct.int64)
    vocab_mask_bool = vocab_mask_1 != ct.full((1,), 0, dtype=ct.bool_)
    vocab_active = vocab_mask_bool & (vocab_idx_scalar >= zero_i) & (vocab_idx_scalar < vocab_rows_i)
    safe_vocab = ct.where(vocab_active, vocab_idx_scalar, vocab_rows_i)
    vocab_row_bcast = ct.astype(safe_vocab, ct.int32) + zero_col
    ct.atomic_add(out_vocab_ptr, (vocab_row_bcast, cols), producer_masked)


@oracle_impl(hardware="B200", point="14a5f5ad", INIT_BLOCK=1024, BLOCK_H=1024)
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1,  # bf16 [50272, 768]
        arg1_1,  # bf16 [16384, 768]
        arg2_1,  # f32 [32, 512, 768]
        arg3_1,  # bf16 [16384, 768]
        arg4_1,  # bf16 [16384, 768]
        arg5_1,  # b8 [32, 512, 768]
        arg6_1,  # f32 [768]
        arg7_1,  # f32 [32, 512, 768]
        arg8_1,  # f32 [32, 512, 1]
        arg9_1,  # i64 [32, 512]
        arg10_1, # i64 [32, 512]
        arg11_1, # b8 [32, 512, 1]
        arg12_1, # i64 [32, 512]
        *_shape_params,
    ) = inputs

    device = arg1_1.device
    # Views
    arg1_2d = arg1_1.view(ROWS, HIDDEN)
    arg2_2d = arg2_1.view(ROWS, HIDDEN)
    arg3_2d = arg3_1.view(ROWS, HIDDEN)
    arg4_2d = arg4_1.view(ROWS, HIDDEN)
    keep_2d = arg5_1.view(ROWS, HIDDEN)
    saved_2d = arg7_1.view(ROWS, HIDDEN)
    scale_1d = arg8_1.view(ROWS)
    position_index_1d = arg9_1.view(ROWS)
    segment_index_1d = arg10_1.view(ROWS)
    vocab_mask_1d = arg11_1.view(ROWS)
    vocab_index_1d = arg12_1.view(ROWS)

    # Outputs
    out_sum3 = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_sum4 = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_position = torch.zeros((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_segment = torch.zeros((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    # Vocab table initialized with a slice of arg0_1 (first VOCAB_ROWS rows) cast to f32
    out_vocab = arg0_1[:VOCAB_ROWS].to(torch.float32).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _row_scatter_reduce_kernel,
        (arg1_2d, arg2_2d, arg3_2d, arg4_2d, keep_2d, arg6_1, saved_2d, scale_1d,
         position_index_1d, segment_index_1d, vocab_mask_1d, vocab_index_1d,
         out_sum3, out_sum4, out_position, out_segment, out_vocab,
         HIDDEN, POSITION_ROWS, SEGMENT_ROWS, VOCAB_ROWS, BLOCK_H_PAD),
    )

    return out_sum3, out_sum4, out_position, out_segment, out_vocab
