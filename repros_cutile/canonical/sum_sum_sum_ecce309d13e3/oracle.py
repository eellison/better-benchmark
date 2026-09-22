"""cuTile port of sum_sum_sum_ecce309d13e3: LayoutLM LN-backward + embedding grad scatter.

Structure mirrors Triton reference's 2-kernel plan:
  Kernel 1 (_init_kernel): initialize dense output tensors — copy bf16 vocab to f32,
     zero the scatter targets, zero the sum3/sum4 accumulators.
  Kernel 2 (_row_scatter_reduce_kernel): per row, compute source = arg1+arg2+arg3+arg4,
     mul1 = source * keep * DROPOUT_SCALE, mul2 = mul1 * weight;
     row_sum, row_dot, producer = row_scale*(mul2*HIDDEN - row_sum - saved*row_dot);
     atomic-add mul1*saved -> out_sum3; mul1 -> out_sum4;
     scatter producer into segment/table_a/table_b/table_c/table_d/position/vocab.

HIDDEN=768 → BLOCK_H=1024 pow2.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
BLOCK_H = 1024
SEGMENT_ROWS = 2
POSITION_ROWS = 512
TABLE_ROWS = 1024
VOCAB_ROWS = 30522
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _init_kernel(
    base_vocab_ptr,   # bf16 (VOCAB_ROWS, HIDDEN) — flattened as (VOCAB_TOTAL,)
    out_sum3_ptr,     # f32 (HIDDEN,)
    out_sum4_ptr,     # f32 (HIDDEN,)
    out_segment_ptr,  # f32 (SEGMENT_ROWS * HIDDEN,)
    out_table_a_ptr,  # f32 (TABLE_ROWS * HIDDEN,)
    out_table_b_ptr,  # f32 (TABLE_ROWS * HIDDEN,)
    out_table_c_ptr,  # f32 (TABLE_ROWS * HIDDEN,)
    out_table_d_ptr,  # f32 (TABLE_ROWS * HIDDEN,)
    out_position_ptr, # f32 (POSITION_ROWS * HIDDEN,)
    out_vocab_ptr,    # f32 (VOCAB_ROWS * HIDDEN,)
    VOCAB_TOTAL: ct.Constant[int],
    TABLE_TOTAL: ct.Constant[int],
    POSITION_TOTAL: ct.Constant[int],
    SEGMENT_TOTAL: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    zero_1d = ct.zeros((BLOCK,), dtype=ct.float32)

    vocab_active = offs < VOCAB_TOTAL
    base = ct.gather(base_vocab_ptr, offs, mask=vocab_active, padding_value=0.0)
    base_f = ct.astype(base, ct.float32)
    ct.scatter(out_vocab_ptr, offs, base_f, mask=vocab_active)

    table_active = offs < TABLE_TOTAL
    ct.scatter(out_table_a_ptr, offs, zero_1d, mask=table_active)
    ct.scatter(out_table_b_ptr, offs, zero_1d, mask=table_active)
    ct.scatter(out_table_c_ptr, offs, zero_1d, mask=table_active)
    ct.scatter(out_table_d_ptr, offs, zero_1d, mask=table_active)

    position_active = offs < POSITION_TOTAL
    segment_active = offs < SEGMENT_TOTAL
    hidden_active = offs < HIDDEN_
    ct.scatter(out_position_ptr, offs, zero_1d, mask=position_active)
    ct.scatter(out_segment_ptr, offs, zero_1d, mask=segment_active)
    ct.scatter(out_sum3_ptr, offs, zero_1d, mask=hidden_active)
    ct.scatter(out_sum4_ptr, offs, zero_1d, mask=hidden_active)


@ct.kernel
def _row_scatter_reduce_kernel(
    arg1_ptr,        # bf16 (ROWS, HIDDEN) view [BATCH,SEQ,HIDDEN]
    arg2_ptr,        # f32 (ROWS, HIDDEN)
    arg3_ptr,        # bf16 (ROWS, HIDDEN)
    arg4_ptr,        # bf16 (ROWS, HIDDEN)
    keep_ptr,        # bool (ROWS, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    saved_ptr,       # f32 (ROWS, HIDDEN)
    row_scale_ptr,   # f32 (ROWS,)
    segment_index_ptr,   # i64 (ROWS,)
    table_a_index_ptr,   # i64 (ROWS,)
    table_b_index_ptr,   # i64 (ROWS,)
    table_c_index_ptr,   # i64 (ROWS,) — flat from [BATCH,SEQ] with alias strides
    table_d_index_ptr,   # i64 (ROWS,) — same
    table_c1_index_ptr,  # i64 (ROWS,)
    table_d1_index_ptr,  # i64 (ROWS,)
    position_index_ptr,  # i64 (SEQ,)
    vocab_index_ptr,     # i64 (ROWS,)
    out_sum3_ptr,        # f32 (HIDDEN,)
    out_sum4_ptr,        # f32 (HIDDEN,)
    out_segment_ptr,     # f32 (SEGMENT_ROWS * HIDDEN,)
    out_table_a_ptr,     # f32 (TABLE_ROWS * HIDDEN,)
    out_table_b_ptr,     # f32 (TABLE_ROWS * HIDDEN,)
    out_table_c_ptr,     # f32 (TABLE_ROWS * HIDDEN,)
    out_table_d_ptr,     # f32 (TABLE_ROWS * HIDDEN,)
    out_position_ptr,    # f32 (POSITION_ROWS * HIDDEN,)
    out_vocab_ptr,       # f32 (VOCAB_ROWS * HIDDEN,)
    HIDDEN_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    h = ct.arange(BLOCK_H_, dtype=ct.int32)
    active = h < HIDDEN_
    zero_1d = ct.zeros((BLOCK_H_,), dtype=ct.float32)

    a1 = ct.astype(
        ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    a1_1d = ct.reshape(a1, (BLOCK_H_,))
    a2 = ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    a2_1d = ct.reshape(a2, (BLOCK_H_,))
    a3 = ct.astype(
        ct.load(arg3_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    a3_1d = ct.reshape(a3, (BLOCK_H_,))
    a4 = ct.astype(
        ct.load(arg4_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    a4_1d = ct.reshape(a4, (BLOCK_H_,))
    source = a2_1d + a1_1d + a3_1d + a4_1d

    keep = ct.astype(
        ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    keep_1d = ct.reshape(keep, (BLOCK_H_,))
    mul1 = source * (keep_1d * DROPOUT_SCALE)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.where(active, weight, zero_1d)
    mul2 = mul1 * weight
    saved = ct.load(saved_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                    padding_mode=ct.PaddingMode.ZERO)
    saved_1d = ct.reshape(saved, (BLOCK_H_,))

    mul2_masked = ct.where(active, mul2, zero_1d)
    mul2_saved = mul2 * saved_1d
    mul2_saved_masked = ct.where(active, mul2_saved, zero_1d)
    row_sum = ct.sum(mul2_masked)  # scalar
    row_dot = ct.sum(mul2_saved_masked)  # scalar

    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_scale_s = ct.reshape(row_scale, ())
    producer = row_scale_s * (mul2 * float(HIDDEN_) - row_sum - saved_1d * row_dot)

    # sum3 and sum4 -> atomic add to (HIDDEN,).
    ct.atomic_add(out_sum3_ptr, (h,), ct.where(active, mul1 * saved_1d, zero_1d))
    ct.atomic_add(out_sum4_ptr, (h,), ct.where(active, mul1, zero_1d))

    zero_scalar = ct.zeros((1,), dtype=ct.int64)

    # Segment: index clamped [0, SEGMENT_ROWS-1].
    segment_raw = ct.gather(segment_index_ptr, ct.full((1,), row, dtype=ct.int32),
                            padding_value=0)
    seg_zero = ct.zeros((1,), dtype=ct.int64)
    seg_max = ct.full((1,), SEGMENT_ROWS - 1, dtype=ct.int64)
    seg_lo = ct.where(segment_raw >= 0, segment_raw, seg_zero)
    seg_idx = ct.where(seg_lo < SEGMENT_ROWS, seg_lo, seg_max)
    seg_off_1d = ct.reshape(seg_idx, (1,)) * HIDDEN_ + h
    seg_active = active
    ct.atomic_add(out_segment_ptr, (seg_off_1d,), ct.where(seg_active, producer, zero_1d))

    # table_a
    table_a_raw = ct.gather(table_a_index_ptr, ct.full((1,), row, dtype=ct.int32),
                            padding_value=-1)
    table_a_ok_s = (table_a_raw >= 0) & (table_a_raw < TABLE_ROWS) & (table_a_raw != -1)
    safe_a = ct.where(table_a_ok_s, table_a_raw, zero_scalar)
    a_off = ct.reshape(safe_a, (1,)) * HIDDEN_ + h
    a_active = active & ct.reshape(table_a_ok_s, (1,))
    ct.atomic_add(out_table_a_ptr, (a_off,), ct.where(a_active, producer, zero_1d))

    # table_b
    table_b_raw = ct.gather(table_b_index_ptr, ct.full((1,), row, dtype=ct.int32),
                            padding_value=-1)
    table_b_ok_s = (table_b_raw >= 0) & (table_b_raw < TABLE_ROWS) & (table_b_raw != -1)
    safe_b = ct.where(table_b_ok_s, table_b_raw, zero_scalar)
    b_off = ct.reshape(safe_b, (1,)) * HIDDEN_ + h
    b_active = active & ct.reshape(table_b_ok_s, (1,))
    ct.atomic_add(out_table_b_ptr, (b_off,), ct.where(b_active, producer, zero_1d))

    # table_c (from arg12 alias-strided; we flatten pre-launch to ROWS-length)
    table_c_raw = ct.gather(table_c_index_ptr, ct.full((1,), row, dtype=ct.int32),
                            padding_value=-1)
    table_c_ok_s = (table_c_raw >= 0) & (table_c_raw < TABLE_ROWS) & (table_c_raw != -1)
    safe_c = ct.where(table_c_ok_s, table_c_raw, zero_scalar)
    c_off = ct.reshape(safe_c, (1,)) * HIDDEN_ + h
    c_active = active & ct.reshape(table_c_ok_s, (1,))
    ct.atomic_add(out_table_c_ptr, (c_off,), ct.where(c_active, producer, zero_1d))

    # table_c1 -> also into out_table_c
    table_c1_raw = ct.gather(table_c1_index_ptr, ct.full((1,), row, dtype=ct.int32),
                             padding_value=-1)
    table_c1_ok_s = (table_c1_raw >= 0) & (table_c1_raw < TABLE_ROWS) & (table_c1_raw != -1)
    safe_c1 = ct.where(table_c1_ok_s, table_c1_raw, zero_scalar)
    c1_off = ct.reshape(safe_c1, (1,)) * HIDDEN_ + h
    c1_active = active & ct.reshape(table_c1_ok_s, (1,))
    ct.atomic_add(out_table_c_ptr, (c1_off,), ct.where(c1_active, producer, zero_1d))

    # table_d
    table_d_raw = ct.gather(table_d_index_ptr, ct.full((1,), row, dtype=ct.int32),
                            padding_value=-1)
    table_d_ok_s = (table_d_raw >= 0) & (table_d_raw < TABLE_ROWS) & (table_d_raw != -1)
    safe_d = ct.where(table_d_ok_s, table_d_raw, zero_scalar)
    d_off = ct.reshape(safe_d, (1,)) * HIDDEN_ + h
    d_active = active & ct.reshape(table_d_ok_s, (1,))
    ct.atomic_add(out_table_d_ptr, (d_off,), ct.where(d_active, producer, zero_1d))

    # table_d1
    table_d1_raw = ct.gather(table_d1_index_ptr, ct.full((1,), row, dtype=ct.int32),
                             padding_value=-1)
    table_d1_ok_s = (table_d1_raw >= 0) & (table_d1_raw < TABLE_ROWS) & (table_d1_raw != -1)
    safe_d1 = ct.where(table_d1_ok_s, table_d1_raw, zero_scalar)
    d1_off = ct.reshape(safe_d1, (1,)) * HIDDEN_ + h
    d1_active = active & ct.reshape(table_d1_ok_s, (1,))
    ct.atomic_add(out_table_d_ptr, (d1_off,), ct.where(d1_active, producer, zero_1d))

    # position (indexed by seq = row % SEQ)
    seq_idx = row - (row // SEQ_) * SEQ_
    pos_raw = ct.gather(position_index_ptr, ct.full((1,), seq_idx, dtype=ct.int32),
                        padding_value=-1)
    pos_ok_s = (pos_raw >= 0) & (pos_raw < POSITION_ROWS) & (pos_raw != -1)
    safe_pos = ct.where(pos_ok_s, pos_raw, zero_scalar)
    pos_off = ct.reshape(safe_pos, (1,)) * HIDDEN_ + h
    pos_active = active & ct.reshape(pos_ok_s, (1,))
    ct.atomic_add(out_position_ptr, (pos_off,), ct.where(pos_active, producer, zero_1d))

    # vocab (skip idx=0)
    vocab_raw = ct.gather(vocab_index_ptr, ct.full((1,), row, dtype=ct.int32),
                          padding_value=0)
    vocab_ok_s = (vocab_raw >= 0) & (vocab_raw < VOCAB_ROWS) & (vocab_raw != 0)
    safe_vocab = ct.where(vocab_ok_s, vocab_raw, zero_scalar)
    vocab_off = ct.reshape(safe_vocab, (1,)) * HIDDEN_ + h
    vocab_active = active & ct.reshape(vocab_ok_s, (1,))
    ct.atomic_add(out_vocab_ptr, (vocab_off,), ct.where(vocab_active, producer, zero_1d))


@oracle_impl(hardware="B200", point="1bcea4a2", INIT_BLOCK=1024, BLOCK_H=1024)
def oracle_forward(inputs, *, INIT_BLOCK=None, BLOCK_H=None):
    (
        arg0_1,  # bf16 vocab base [VOCAB, HIDDEN]
        arg1_1,  # bf16 [BATCH, SEQ, HIDDEN] add-source
        arg2_1,  # f32 [BATCH, SEQ, HIDDEN]  base
        arg3_1,  # bf16 [BATCH, SEQ, HIDDEN]
        arg4_1,  # bf16 [BATCH, SEQ, HIDDEN]
        arg5_1,  # bool [BATCH, SEQ, HIDDEN]
        arg6_1,  # f32 [HIDDEN] weight
        arg7_1,  # f32 [BATCH, SEQ, HIDDEN]  saved_normed
        arg8_1,  # f32 [BATCH, SEQ, 1]        row_scale (strided)
        arg9_1,  # i64 segment index [BATCH, SEQ]
        arg10_1, # i64 table_a index [BATCH, SEQ]
        arg11_1, # i64 table_b index [BATCH, SEQ]
        arg12_1, # i64 table_c index [BATCH, SEQ] with alias strides
        arg13_1, # i64 table_d index [BATCH, SEQ] alias
        arg14_1, # i64 table_c1 index [BATCH, SEQ] alias
        arg15_1, # i64 table_d1 index [BATCH, SEQ] alias
        arg16_1, # i64 position index [SEQ]
        arg17_1, # i64 vocab index [BATCH, SEQ]
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    hidden = HIDDEN

    # Prepare 2D views (ROWS, HIDDEN).
    rows = ROWS
    arg1_2d = arg1_1.view(rows, hidden)
    arg2_2d = arg2_1.view(rows, hidden)
    arg3_2d = arg3_1.view(rows, hidden)
    arg4_2d = arg4_1.view(rows, hidden)
    keep_2d = arg5_1.view(rows, hidden)
    saved_2d = arg7_1.view(rows, hidden)
    row_scale_1d = arg8_1.reshape(rows).contiguous()

    # Segment index -> [BATCH*SEQ] flat.
    seg_idx_1d = arg9_1.reshape(rows).contiguous()
    a_idx_1d = arg10_1.reshape(rows).contiguous()
    b_idx_1d = arg11_1.reshape(rows).contiguous()
    # arg12_1, arg13_1, arg14_1, arg15_1 come as [BATCH, SEQ] alias-strided.
    # Materialize dense (ROWS,) copies (matches ALIAS_S0/S1 indexing).
    c_idx_1d = arg12_1.reshape(BATCH, SEQ).contiguous().view(rows)
    d_idx_1d = arg13_1.reshape(BATCH, SEQ).contiguous().view(rows)
    c1_idx_1d = arg14_1.reshape(BATCH, SEQ).contiguous().view(rows)
    d1_idx_1d = arg15_1.reshape(BATCH, SEQ).contiguous().view(rows)
    pos_idx_1d = arg16_1.reshape(SEQ).contiguous()
    vocab_idx_1d = arg17_1.reshape(rows).contiguous()

    # Output allocations.
    out_sum3 = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, hidden), device=device, dtype=torch.float32)
    out_table_a = torch.empty((TABLE_ROWS, hidden), device=device, dtype=torch.float32)
    out_table_b = torch.empty((TABLE_ROWS, hidden), device=device, dtype=torch.float32)
    out_table_c = torch.empty((TABLE_ROWS, hidden), device=device, dtype=torch.float32)
    out_table_d = torch.empty((TABLE_ROWS, hidden), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, hidden), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, hidden), device=device, dtype=torch.float32)

    vocab_total = VOCAB_ROWS * hidden
    table_total = TABLE_ROWS * hidden
    position_total = POSITION_ROWS * hidden
    segment_total = SEGMENT_ROWS * hidden

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        ((vocab_total + INIT_BLOCK - 1) // INIT_BLOCK, 1, 1),
        _init_kernel,
        (
            arg0_1.reshape(-1),
            out_sum3, out_sum4,
            out_segment.reshape(-1),
            out_table_a.reshape(-1), out_table_b.reshape(-1),
            out_table_c.reshape(-1), out_table_d.reshape(-1),
            out_position.reshape(-1), out_vocab.reshape(-1),
            vocab_total, table_total, position_total, segment_total,
            hidden, INIT_BLOCK,
        ),
    )
    ct.launch(
        stream,
        (rows, 1, 1),
        _row_scatter_reduce_kernel,
        (
            arg1_2d, arg2_2d, arg3_2d, arg4_2d, keep_2d,
            arg6_1, saved_2d, row_scale_1d,
            seg_idx_1d, a_idx_1d, b_idx_1d,
            c_idx_1d, d_idx_1d, c1_idx_1d, d1_idx_1d,
            pos_idx_1d, vocab_idx_1d,
            out_sum3, out_sum4,
            out_segment.reshape(-1),
            out_table_a.reshape(-1), out_table_b.reshape(-1),
            out_table_c.reshape(-1), out_table_d.reshape(-1),
            out_position.reshape(-1), out_vocab.reshape(-1),
            hidden, SEQ, BLOCK_H,
        ),
    )

    return (
        out_sum3, out_sum4,
        out_segment,
        out_table_a, out_table_b, out_table_c, out_table_d,
        out_position, out_vocab,
    )
