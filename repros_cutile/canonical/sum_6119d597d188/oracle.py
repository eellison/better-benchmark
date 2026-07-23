"""cuTile port of sum_6119d597d188: PLBart LM-head softmax backward.

Computes: for each (row, col) in [ROWS=16384, COLS=50005]:
  probs = exp(logits[r, c] - row_shift[r] - row_logsum[r])  (bf16 rounded)
  neg_scale = -numerator/denominator (bf16 rounded, then f32)
  target = (col == labels[r] AND label != -100) ? neg_scale : 0
  row_sum = (label != -100 AND label in [0, COLS)) ? neg_scale : 0
  softmax_grad = target - probs * row_sum        (bf16 rounded)
  out[r, c] = residual[r, c] + softmax_grad      (bf16 rounded)

Both output layouts (row and transposed) get the same values, padded from
COLS=50005 to PADDED_COLS=50008 with zeros. We pre-zero the outputs (which
covers the pad columns) then run a cuTile kernel that writes only cols < COLS
via `ct.scatter` with a valid-column mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16 * 1024
COLS = 50005
PADDED_COLS = 50008


@ct.kernel
def _plbart_lm_backward_kernel(
    labels_ptr,          # i64 [ROWS]  (viewed from [16, 1024])
    logits_ptr,          # bf16 [ROWS, PADDED_COLS]  (row stride PADDED_COLS)
    row_shift_ptr,       # f32 [ROWS]
    row_logsum_ptr,      # f32 [ROWS]
    residual_ptr,        # bf16 [ROWS, COLS]
    neg_scale_row_ptr,   # f32 [ROWS]  (= -scale where label != -100 else 0, bf16 rounded)
    out_row_ptr,         # bf16 [ROWS, PADDED_COLS]  (row stride PADDED_COLS)
    out_transposed_ptr,  # bf16 [PADDED_COLS, ROWS]  (row stride ROWS)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    N_COLS: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)

    rows = row_tile * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int64)
    cols = col_tile * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int64)
    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    col_valid = cols < N_COLS  # (BLOCK_N,)
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_N))

    labels = ct.load(labels_ptr, index=(row_tile,), shape=(BLOCK_M,))
    labels_2d = ct.reshape(labels, (BLOCK_M, 1))
    label_active = labels_2d != -100  # (BLOCK_M, 1)
    label_active_broadcast = ct.reshape(
        ct.astype(label_active, ct.int32) + ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.int32),
        (BLOCK_M, BLOCK_N),
    ) != 0

    neg_scale = ct.load(neg_scale_row_ptr, index=(row_tile,), shape=(BLOCK_M,))  # f32
    neg_scale_2d = ct.reshape(neg_scale, (BLOCK_M, 1))

    # target_grad: (col == label) & label_active → neg_scale else 0
    is_target = (cols_2d == labels_2d) & label_active
    zero_f = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    target_grad = ct.where(is_target, neg_scale_2d + zero_f, zero_f)

    # row_sum: label_active → neg_scale else 0. Broadcast to (BLOCK_M, BLOCK_N).
    row_sum = neg_scale_2d + zero_f

    # Load logits with row-stride PADDED_COLS (real array shape (ROWS, PADDED_COLS))
    # We tile over (BLOCK_M, BLOCK_N). Cols may exceed COLS but stay in PADDED_COLS.
    # Since logits has PADDED_COLS row stride, we can safely load cols up to PADDED_COLS.
    logits = ct.load(
        logits_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits_f = ct.astype(logits, ct.float32)
    row_shift = ct.load(row_shift_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_logsum = ct.load(row_logsum_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_shift_2d = ct.reshape(row_shift, (BLOCK_M, 1))
    row_logsum_2d = ct.reshape(row_logsum, (BLOCK_M, 1))

    shifted = logits_f - row_shift_2d - row_logsum_2d
    shifted_bf16 = ct.astype(shifted, ct.bfloat16)
    probs = ct.exp(ct.astype(shifted_bf16, ct.float32))

    correction = target_grad - probs * row_sum
    correction_bf16 = ct.astype(correction, ct.bfloat16)

    # Load residual — has shape (ROWS, COLS), row stride COLS. cols may go beyond COLS;
    # use padding_mode=ZERO for the OOB reads (they'll be masked out on store).
    residual = ct.load(
        residual_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    out_val_f = ct.astype(residual, ct.float32) + ct.astype(correction_bf16, ct.float32)
    out_val = ct.astype(out_val_f, ct.bfloat16)

    # Scatter into both outputs with col mask (col < N_COLS).
    # For out_row: (row, col)
    ct.scatter(
        out_row_ptr,
        (rows_2d, cols_2d),
        out_val,
        mask=col_valid_2d,
    )
    # For out_transposed: (col, row) → flip index tuple
    ct.scatter(
        out_transposed_ptr,
        (cols_2d, rows_2d),
        out_val,
        mask=col_valid_2d,
    )


@oracle_impl(hardware="B200", point="b73ba80f", BLOCK_M=16, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    (
        arg0_1,   # f32 [] numerator
        arg1_1,   # f32 [] denominator
        arg2_1,   # i64 [16, 1024] labels
        arg3_1,   # bf16 [16, 1024, 50005] logits (row stride 50008)
        arg4_1,   # f32 [16384, 1] row_shift
        arg5_1,   # f32 [16384, 1] row_logsum
        arg6_1,   # bf16 [16, 1024, 50005] residual
        *_shape_params,
    ) = inputs
    device = arg3_1.device

    # Zero-init both padded outputs — covers the [COLS, PADDED_COLS) region for free.
    out_row = torch.zeros(
        (ROWS, PADDED_COLS), device=device, dtype=torch.bfloat16,
    )
    out_transposed = torch.zeros(
        (PADDED_COLS, ROWS), device=device, dtype=torch.bfloat16,
    )

    # Compute -scale in torch (bf16 rounded then to f32).
    scale = (arg0_1 / arg1_1).to(torch.bfloat16).to(torch.float32)
    neg_scale_scalar = -scale

    # Compute per-row neg_scale: where label != -100 else 0 (bf16 rounded).
    labels_flat = arg2_1.view(ROWS)
    label_active = labels_flat != -100
    neg_scale_row = torch.where(
        label_active,
        neg_scale_scalar.expand(ROWS),
        torch.zeros((), device=device, dtype=torch.float32).expand(ROWS),
    )
    # Round-trip bf16 to match the Triton oracle's rounding path.
    neg_scale_row = neg_scale_row.to(torch.bfloat16).to(torch.float32).contiguous()

    # Prepare views. arg3_1 has strides (51208192, 50008, 1) with shape (16, 1024, 50005).
    # View as (ROWS, COLS) using its actual (PADDED_COLS-stride) layout — this is a
    # valid view. The storage supports COLS-wide loads with PADDED_COLS-stride rows.
    logits_2d = torch.as_strided(
        arg3_1,
        (ROWS, COLS),
        (PADDED_COLS, 1),
        storage_offset=arg3_1.storage_offset(),
    )
    residual_2d = arg6_1.view(ROWS, COLS)

    row_shift_1d = arg4_1.view(ROWS)
    row_logsum_1d = arg5_1.view(ROWS)

    stream = torch.cuda.current_stream()
    n_row_tiles = ROWS // BLOCK_M
    n_col_tiles = ct.cdiv(COLS, BLOCK_N)
    ct.launch(
        stream,
        (n_row_tiles, n_col_tiles, 1),
        _plbart_lm_backward_kernel,
        (
            labels_flat, logits_2d, row_shift_1d, row_logsum_1d, residual_2d,
            neg_scale_row, out_row, out_transposed, BLOCK_M, BLOCK_N, COLS,
        ),
    )
    return out_transposed, out_row
