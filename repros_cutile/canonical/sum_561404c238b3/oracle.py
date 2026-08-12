"""cuTile port of sum_561404c238b3: M2M100 bf16 cross-entropy backward.

The algebraic story of the Triton oracle:
  scale = arg0 / arg1              # f32 scalar
  label[8192] = view(arg2, [-1])
  ne = label != -100
  safe_label = where(ne, label, 0)
  # one_hot_f32 = (safe_label == iota_vocab) ? -1 : 0
  # scale_row  = where(ne, scale, 0)
  # `where_1 * where_2` = -scale_row when col == safe_label && col < VOCAB, else 0
  # After bf16 round trip: row_sum_f32 = bf16(-scale_row) at the label col only
  # Sum over cols (i.e. sum_1[row]) = row_sum_f32 if in_vocab & ne else 0

  # exp branch:
  logits_f = f32(arg3.view(rows, VOCAB))
  shifted = f32( bf16( logits_f - arg4 - arg5 ) )
  exp_v = exp(shifted)

  # sub_2 = grad_one_hot_bf16_f32 - exp*sum_1_bcast   (pointwise)
  # delta = bf16(sub_2)
  # out = bf16(arg6 + delta)
  # view_4[rows, vocab], view_4.permute(1, 0)

Since VOCAB=128112 is not a power of two, we pad column-wise inside the kernel
and only write valid columns via `ct.scatter`.

Approach:
  - torch handles the row-scalar chain (row_sum bf16-rounded), which is tiny
    (per-row work; no big allocation).
  - cuTile handles the huge 2D materialization: reads logits & residual,
    computes shifted+bf16 round+exp, computes delta including one-hot label
    subtraction, adds residual, writes bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


VOCAB = 128112
ROWS = 8192


@ct.kernel
def _xent_bwd_kernel(
    logits_ptr,      # bf16 [ROWS, VOCAB]
    residual_ptr,    # bf16 [ROWS, VOCAB]
    row_sum_ptr,     # f32  [ROWS]  (already bf16-rounded scalar per row)
    shift0_ptr,      # f32  [ROWS]
    shift1_ptr,      # f32  [ROWS]
    safe_label_ptr,  # i64  [ROWS]
    out_ptr,         # bf16 [ROWS, VOCAB]
    ROWS_N: ct.Constant[int],
    VOCAB_N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)
    cols = col_block * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask = cols < VOCAB_N

    # Per-row scalars
    row_sum = ct.load(row_sum_ptr, index=(row,), shape=(1,))
    shift0 = ct.astype(ct.load(shift0_ptr, index=(row,), shape=(1,)), ct.float32)
    shift1 = ct.astype(ct.load(shift1_ptr, index=(row,), shape=(1,)), ct.float32)
    safe_label = ct.load(safe_label_ptr, index=(row,), shape=(1,))
    row_sum_1d = ct.reshape(row_sum, (1,))
    shift0_1d = ct.reshape(shift0, (1,))
    shift1_1d = ct.reshape(shift1, (1,))
    safe_label_1d = ct.reshape(safe_label, (1,))

    # Load logits (bf16), residual (bf16) with ZERO padding
    logits = ct.load(logits_ptr, index=(row, col_block), shape=(1, BLOCK_N),
                     padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, col_block), shape=(1, BLOCK_N),
                       padding_mode=ct.PaddingMode.ZERO)
    logits_f = ct.astype(logits, ct.float32)
    logits_1d = ct.reshape(logits_f, (BLOCK_N,))
    shifted = logits_1d - shift0_1d - shift1_1d
    shifted_bf = ct.astype(shifted, ct.bfloat16)
    shifted_f = ct.astype(shifted_bf, ct.float32)
    exp_v = ct.exp(shifted_f)

    # grad_one_hot_f32 per element: -bf16(scale) when col == safe_label && col<VOCAB & ne
    # But row_sum already encodes this: it's already bf16(-scale) if active,
    # and 0 otherwise. So the one-hot column contribution is row_sum at
    # col==safe_label, else 0.
    label_col_i32 = ct.astype(safe_label_1d, ct.int32)  # broadcasts to (BLOCK_N,)
    is_label_col = cols == label_col_i32
    zero_f = ct.zeros((BLOCK_N,), dtype=ct.float32)
    one_hot_scaled = ct.where(is_label_col, row_sum_1d, zero_f)

    correction_f = one_hot_scaled - exp_v * row_sum_1d
    correction_bf = ct.astype(correction_f, ct.bfloat16)

    residual_1d = ct.reshape(residual, (BLOCK_N,))
    out_f = ct.astype(residual_1d, ct.float32) + ct.astype(correction_bf, ct.float32)
    out_bf = ct.astype(out_f, ct.bfloat16)

    # Scatter write with column mask (VOCAB non-pow2)
    row_idx = ct.full((BLOCK_N,), row, dtype=ct.int32)
    ct.scatter(out_ptr, (row_idx, cols), out_bf, mask=col_mask)


@oracle_impl(hardware="B200", point="46e5658a")
def oracle_forward(inputs):
    (
        numerator, denominator, labels, logits, shift0, shift1, residual,
        _shape0, _shape1, _shape2, _shape3, _shape4,
    ) = inputs
    device = logits.device
    rows = ROWS
    vocab = VOCAB
    labels_flat = labels.view(-1)

    # Compute scale scalar & per-row scale
    scale = (numerator / denominator).view(())
    ne = labels_flat != -100
    safe_label = torch.where(ne, labels_flat, torch.zeros_like(labels_flat))
    in_vocab = (safe_label >= 0) & (safe_label < vocab)
    active = ne & in_vocab
    # row_sum = bf16(-scale) if active, else 0
    neg_scale_bf = (-scale).to(torch.bfloat16).to(torch.float32)
    row_sum_f = torch.where(active, neg_scale_bf.expand_as(safe_label), torch.zeros_like(labels_flat, dtype=torch.float32))

    # Prepare shift0, shift1 as 1D f32
    shift0_1d = shift0.view(rows).contiguous()
    shift1_1d = shift1.view(rows).contiguous()

    # Prepare logits and residual as 2D contiguous bf16 [rows, vocab]
    logits_2d = logits.view(rows, vocab).contiguous()
    residual_2d = residual.view(rows, vocab).contiguous()

    # Output bf16 [rows, vocab]
    out = torch.empty_strided(
        (rows, vocab), (vocab, 1),
        device=device, dtype=torch.bfloat16,
    )

    # BLOCK_N must be power of 2 covering vocab with padding
    BLOCK_N = 512
    n_col_blocks = (vocab + BLOCK_N - 1) // BLOCK_N  # 251 blocks

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, n_col_blocks, 1), _xent_bwd_kernel,
        (logits_2d, residual_2d, row_sum_f, shift0_1d, shift1_1d, safe_label,
         out, rows, vocab, BLOCK_N),
    )

    return out, out.permute(1, 0)
