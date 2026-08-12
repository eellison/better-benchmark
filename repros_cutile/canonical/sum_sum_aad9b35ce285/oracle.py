"""cuTile port of sum_sum_aad9b35ce285: Longformer masked-LM bf16 CE-backward.

The Triton oracle materializes a bf16 sparse-label CE gradient into a padded
[ROWS, VOCAB_PAD] output and its [VOCAB_PAD, ROWS] transpose, then reduces
column sums (bf16-roundtripped). The inline_asm add.rn/sub.rn/mul.rn PTX
maps to cuTile's default round-to-nearest-even fp32 arithmetic.

Two sequential kernels:
1. `_materialize_kernel` — per (row_block, col_block) tile: compute the sparse
   grad and residual add, then scatter-store into out_rows[ROWS, VOCAB_PAD]
   AND out_trans[VOCAB_PAD, ROWS] with column masks (only writing valid vocab
   positions). Also accumulates per-tile column partials into a
   [num_row_blocks, VOCAB_PAD] partial buffer.
2. `_column_sum_kernel` — sum the partials across row tiles, bf16 roundtrip.

VOCAB=50265 is non-pow2; we iterate the vocab in BLOCK_N=128 tiles and use
scatter with a col-valid mask for stores. Out tensors are pre-zeroed so pad
columns [50265..50271] stay 0 (matches constant_pad_nd semantics).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
VOCAB = 50265
VOCAB_PAD = 50272
PAD = 7


@ct.kernel
def _materialize_kernel(
    numerator_ptr,     # f32 scalar (shape [])
    denominator_ptr,   # f32 scalar (shape [])
    labels_ptr,        # i64 [ROWS]
    logits_ptr,        # bf16 [ROWS, VOCAB_PAD]  (strided so col-vector is contig)
    row_shift0_ptr,    # f32 [ROWS]
    row_shift1_ptr,    # f32 [ROWS]
    residual_ptr,      # bf16 [ROWS, VOCAB]      (contiguous [ROWS, VOCAB])
    out_rows_ptr,      # bf16 [ROWS, VOCAB_PAD]  (pre-zeroed)
    out_trans_ptr,     # bf16 [VOCAB_PAD, ROWS]  (pre-zeroed)
    partial_ptr,       # f32 [NUM_ROW_BLOCKS, VOCAB]   partial column sums
    ROWS_N: ct.Constant[int],
    VOCAB_N: ct.Constant[int],
    VOCAB_PAD_N: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    rows = ct.arange(BLOCK_M, dtype=ct.int32) + row_block * BLOCK_M
    cols = ct.arange(BLOCK_N, dtype=ct.int32) + col_block * BLOCK_N
    row_valid = rows < ROWS_N
    col_valid = cols < VOCAB_N
    row_valid_2d = ct.reshape(row_valid, (BLOCK_M, 1))
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_N))
    mask2 = ct.broadcast_to(row_valid_2d, (BLOCK_M, BLOCK_N)) & ct.broadcast_to(
        col_valid_2d, (BLOCK_M, BLOCK_N)
    )

    # Row-level scalars.
    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    zero_i32 = ct.zeros((BLOCK_M,), dtype=ct.int32)
    zero_i64 = ct.full((BLOCK_M,), 0, dtype=ct.int64)
    safe_row = ct.where(row_valid, rows, zero_i32)

    label = ct.gather(labels_ptr, (safe_row,), mask=row_valid, padding_value=-100)
    minus100 = ct.full((BLOCK_M,), -100, dtype=ct.int64)
    active_label = label != minus100
    safe_label = ct.where(active_label, label, zero_i64)
    vocab_lo = ct.zeros((BLOCK_M,), dtype=ct.int64)
    vocab_hi = ct.full((BLOCK_M,), VOCAB_N, dtype=ct.int64)
    in_vocab = (safe_label >= vocab_lo) & (safe_label < vocab_hi)

    numerator = ct.astype(ct.load(numerator_ptr, index=(0,), shape=(1,)), ct.float32)
    denominator = ct.astype(
        ct.load(denominator_ptr, index=(0,), shape=(1,)), ct.float32
    )
    scale_s = ct.reshape(numerator / denominator, ())
    # row_sum: negated scale, rounded through bf16 (matches Triton bf16 roundtrip)
    neg_scale = -scale_s
    neg_scale_bf = ct.astype(neg_scale, ct.bfloat16)
    neg_scale_f = ct.astype(neg_scale_bf, ct.float32)
    zero_f_row = ct.zeros((BLOCK_M,), dtype=ct.float32)
    neg_scale_row = ct.full((BLOCK_M,), 0.0, dtype=ct.float32) + neg_scale_f
    row_sum = ct.where(active_label & in_vocab, neg_scale_row, zero_f_row)

    # one_hot: safe_label[i] == cols[j]
    safe_label_i32 = ct.astype(safe_label, ct.int32)
    safe_label_2d = ct.reshape(safe_label_i32, (BLOCK_M, 1))
    cols_bcast = ct.reshape(cols, (1, BLOCK_N))
    one_hot = ct.broadcast_to(safe_label_2d, (BLOCK_M, BLOCK_N)) == ct.broadcast_to(
        cols_bcast, (BLOCK_M, BLOCK_N)
    )
    active_label_2d = ct.broadcast_to(ct.reshape(active_label, (BLOCK_M, 1)),
                                       (BLOCK_M, BLOCK_N))
    zero2_f = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    row_sum_2d = ct.broadcast_to(ct.reshape(row_sum, (BLOCK_M, 1)),
                                  (BLOCK_M, BLOCK_N))
    grad_f32 = ct.where(one_hot & active_label_2d, row_sum_2d, zero2_f)

    # Load logits[rows, cols] using gather with mask (VOCAB_PAD stride so col cols is valid slot).
    rows_broad = ct.broadcast_to(rows_2d, (BLOCK_M, BLOCK_N))
    cols_broad = ct.broadcast_to(cols_2d, (BLOCK_M, BLOCK_N))
    zero_i32_2d = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.int32)
    safe_rows = ct.where(mask2, rows_broad, zero_i32_2d)
    safe_cols = ct.where(mask2, cols_broad, zero_i32_2d)

    logits_bf = ct.gather(logits_ptr, (safe_rows, safe_cols),
                          mask=mask2, padding_value=0)
    logits_f = ct.astype(logits_bf, ct.float32)

    row_shift0 = ct.astype(
        ct.gather(row_shift0_ptr, (safe_row,), mask=row_valid, padding_value=0),
        ct.float32,
    )
    row_shift1 = ct.astype(
        ct.gather(row_shift1_ptr, (safe_row,), mask=row_valid, padding_value=0),
        ct.float32,
    )
    row_shift0_2d = ct.broadcast_to(ct.reshape(row_shift0, (BLOCK_M, 1)),
                                    (BLOCK_M, BLOCK_N))
    row_shift1_2d = ct.broadcast_to(ct.reshape(row_shift1, (BLOCK_M, 1)),
                                    (BLOCK_M, BLOCK_N))

    centered_f = logits_f - row_shift0_2d - row_shift1_2d
    # bf16 roundtrip boundary
    centered_bf = ct.astype(centered_f, ct.bfloat16)
    centered_f = ct.astype(centered_bf, ct.float32)
    exp_values = ct.exp(centered_f)

    correction = grad_f32 - exp_values * row_sum_2d
    correction_bf = ct.astype(correction, ct.bfloat16)

    residual = ct.gather(residual_ptr, (safe_rows, safe_cols),
                         mask=mask2, padding_value=0)
    out_value_f = ct.astype(residual, ct.float32) + ct.astype(correction_bf, ct.float32)
    out_value = ct.astype(out_value_f, ct.bfloat16)

    # Store into pre-zeroed padded outputs, masked to (row_valid & col_valid).
    ct.scatter(out_rows_ptr, (safe_rows, safe_cols), out_value, mask=mask2)
    # Transposed output has stride (ROWS, 1): index (col, row).
    ct.scatter(out_trans_ptr, (safe_cols, safe_rows), out_value, mask=mask2)

    # Per-tile column partial sums (f32) — match torch's sum(view_4, dim=0,
    # dtype=f32): each bf16 element is upcast to f32 before addition.
    out_value_bf_f = ct.astype(out_value, ct.float32)
    out_value_bf_f_masked = ct.where(mask2, out_value_bf_f, zero2_f)
    partial = ct.sum(out_value_bf_f_masked, axis=0)
    row_block_tile = ct.full((BLOCK_N,), row_block, dtype=ct.int32)
    ct.scatter(partial_ptr, (row_block_tile, cols), partial, mask=col_valid)


@ct.kernel
def _column_sum_kernel(
    partial_ptr,       # f32 [NUM_ROW_BLOCKS, VOCAB]
    out_sum_ptr,       # f32 [VOCAB]
    NUM_ROW_BLOCKS: ct.Constant[int],
    VOCAB_N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    col_block = ct.bid(0)
    cols = ct.arange(BLOCK_N, dtype=ct.int32) + col_block * BLOCK_N
    col_valid = cols < VOCAB_N

    acc = ct.zeros((BLOCK_N,), dtype=ct.float32)
    num_iters = (NUM_ROW_BLOCKS + BLOCK_R - 1) // BLOCK_R
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    for i in range(num_iters):
        rows = ct.arange(BLOCK_R, dtype=ct.int32) + i * BLOCK_R
        row_valid = rows < NUM_ROW_BLOCKS
        row_valid_2d = ct.reshape(row_valid, (BLOCK_R, 1))
        col_valid_2d = ct.reshape(col_valid, (1, BLOCK_N))
        mask2 = ct.broadcast_to(row_valid_2d, (BLOCK_R, BLOCK_N)) & ct.broadcast_to(
            col_valid_2d, (BLOCK_R, BLOCK_N)
        )
        rows_broad = ct.broadcast_to(ct.reshape(rows, (BLOCK_R, 1)),
                                      (BLOCK_R, BLOCK_N))
        cols_broad = ct.broadcast_to(cols_2d, (BLOCK_R, BLOCK_N))
        zero_i32_2d = ct.zeros((BLOCK_R, BLOCK_N), dtype=ct.int32)
        safe_rows = ct.where(mask2, rows_broad, zero_i32_2d)
        safe_cols = ct.where(mask2, cols_broad, zero_i32_2d)
        vals = ct.gather(partial_ptr, (safe_rows, safe_cols),
                         mask=mask2, padding_value=0)
        zero2 = ct.zeros((BLOCK_R, BLOCK_N), dtype=ct.float32)
        acc = acc + ct.sum(ct.where(mask2, vals, zero2), axis=0)

    # bf16 roundtrip on the column sum (matches Triton view_5 -> bf16 -> f32 cast).
    acc_bf = ct.astype(acc, ct.bfloat16)
    acc_f = ct.astype(acc_bf, ct.float32)
    ct.scatter(out_sum_ptr, (cols,), acc_f, mask=col_valid)


@oracle_impl(
    hardware="B200",
    point="5599a41a",
    BLOCK_M=16,
    BLOCK_N=128,
    FINAL_BLOCK_N=128,
    FINAL_BLOCK_R=64,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
    FINAL_BLOCK_R: int,
):
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        _shape0,
        _shape1,
        _shape2,
        _shape3,
        _shape4,
        _shape5,
        _shape6,
        _shape7,
    ) = inputs

    device = logits.device
    # logits comes in as bf16 [8, 1024, 50265] with stride (51478528, 50272, 1);
    # a [ROWS, VOCAB] view over the same storage keeps the stride (VOCAB_PAD, 1) —
    # last-dim contiguous, second-dim = VOCAB_PAD.
    logits_2d = logits.view(ROWS, VOCAB)
    labels_1d = labels.view(ROWS).contiguous()
    row_shift0_1d = row_shift0.view(ROWS).contiguous()
    row_shift1_1d = row_shift1.view(ROWS).contiguous()
    # residual is bf16 [8, 1024, 50265] contiguous. View as [ROWS, VOCAB].
    residual_2d = residual.view(ROWS, VOCAB)
    # Scalar tensors (rank 0) — cuTile can't index rank-0 arrays; reshape to [1].
    numerator_1d = numerator.reshape(1)
    denominator_1d = denominator.reshape(1)

    # Pre-zero the padded outputs so scatter with col-mask leaves pad cols = 0.
    out_rows = torch.zeros(
        (ROWS, VOCAB_PAD), device=device, dtype=torch.bfloat16,
    )
    out_trans = torch.zeros(
        (VOCAB_PAD, ROWS), device=device, dtype=torch.bfloat16,
    )
    out_sum = torch.empty((VOCAB,), device=device, dtype=torch.float32)

    num_row_blocks = (ROWS + BLOCK_M - 1) // BLOCK_M
    num_col_blocks = (VOCAB + BLOCK_N - 1) // BLOCK_N
    partial = torch.zeros(
        (num_row_blocks, VOCAB), device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_blocks, num_col_blocks, 1),
        _materialize_kernel,
        (
            numerator_1d,
            denominator_1d,
            labels_1d,
            logits_2d,
            row_shift0_1d,
            row_shift1_1d,
            residual_2d,
            out_rows,
            out_trans,
            partial,
            ROWS,
            VOCAB,
            VOCAB_PAD,
            BLOCK_M,
            BLOCK_N,
        ),
    )
    ct.launch(
        stream,
        ((VOCAB + FINAL_BLOCK_N - 1) // FINAL_BLOCK_N, 1, 1),
        _column_sum_kernel,
        (
            partial,
            out_sum,
            num_row_blocks,
            VOCAB,
            FINAL_BLOCK_N,
            FINAL_BLOCK_R,
        ),
    )

    return out_rows, out_trans, out_sum
