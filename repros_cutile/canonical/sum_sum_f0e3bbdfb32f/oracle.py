"""cuTile port of sum_sum_f0e3bbdfb32f: DeBERTa xent-backward with column sum.

Similar to sum_561404c238b3 (M2M100) but with three outputs:
  1. constant_pad_nd:   bf16 [rows, VOCAB_PAD]  (right-padded row-major)
  2. constant_pad_nd_1: bf16 [VOCAB_PAD, rows]  (right-padded transpose)
  3. convert_element_type_7: f32 [VOCAB]  = bf16-to-f32 of bf16-rounded
     f32-column-sum of view_4[rows, VOCAB]

Where VOCAB=128100, VOCAB_PAD=128104 (VOCAB rounded up to be aligned).

Approach:
  - cuTile kernel writes the full bf16 delta+residual into a padded
    [rows, VOCAB_PAD] buffer (with cols in [VOCAB, VOCAB_PAD) explicitly
    zeroed), so the row-padded output is directly the output tile.
  - We compute the transpose in torch (it's just a stride reinterpretation).
  - Column sum via a second cuTile kernel over the bf16 rows.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 4096
VOCAB = 128100
VOCAB_PAD = 128104


@ct.kernel
def _xent_bwd_kernel(
    logits_ptr,      # bf16 [ROWS, VOCAB_PAD]  (input has stride VOCAB_PAD)
    residual_ptr,    # bf16 [ROWS, VOCAB]
    row_sum_ptr,     # f32  [ROWS]
    shift0_ptr,      # f32  [ROWS]
    shift1_ptr,      # f32  [ROWS]
    safe_label_ptr,  # i64  [ROWS]
    out_padded_ptr,  # bf16 [ROWS, VOCAB_PAD]  (output row-padded)
    out_trans_ptr,   # bf16 [VOCAB_PAD, ROWS]  (output transpose-padded)
    ROWS_N: ct.Constant[int],
    VOCAB_N: ct.Constant[int],
    VOCAB_PAD_N: ct.Constant[int],
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

    label_col_i32 = ct.astype(safe_label_1d, ct.int32)
    is_label_col = cols == label_col_i32
    zero_f = ct.zeros((BLOCK_N,), dtype=ct.float32)
    one_hot_scaled = ct.where(is_label_col, row_sum_1d, zero_f)

    correction_f = one_hot_scaled - exp_v * row_sum_1d
    correction_bf = ct.astype(correction_f, ct.bfloat16)

    residual_1d = ct.reshape(residual, (BLOCK_N,))
    out_f = ct.astype(residual_1d, ct.float32) + ct.astype(correction_bf, ct.float32)
    zero_bf = ct.full((BLOCK_N,), 0.0, dtype=ct.bfloat16)
    # Write valid columns; padded columns (VOCAB..VOCAB_PAD) get zero via scatter mask.
    # We use two writes: one for valid cols, one for padded cols.
    out_bf_valid = ct.where(col_mask, ct.astype(out_f, ct.bfloat16), zero_bf)

    row_idx = ct.full((BLOCK_N,), row, dtype=ct.int32)
    # Row-padded output: write ALL cols in this block; padded cols get 0.
    # Also need to ensure cols >= VOCAB_PAD don't get written (would be OOB).
    pad_mask = cols < VOCAB_PAD_N
    ct.scatter(out_padded_ptr, (row_idx, cols), out_bf_valid, mask=pad_mask)
    # Transpose output: index (col, row)
    col_idx = cols
    ct.scatter(out_trans_ptr, (col_idx, row_idx), out_bf_valid, mask=pad_mask)


@ct.kernel
def _column_sum_kernel(
    src_ptr,          # bf16 [ROWS, VOCAB_PAD]
    partial_ptr,      # f32  [num_row_tiles, VOCAB]
    ROW_BLOCK: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
    VOCAB_N: ct.Constant[int],
    VOCAB_PAD_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    cols = col_tile * COL_BLOCK + ct.arange(COL_BLOCK, dtype=ct.int32)
    col_mask = cols < VOCAB_N
    # Load [ROW_BLOCK, COL_BLOCK] from bf16 [ROWS, VOCAB_PAD]
    values = ct.load(src_ptr, index=(row_tile, col_tile),
                     shape=(ROW_BLOCK, COL_BLOCK),
                     padding_mode=ct.PaddingMode.ZERO)
    values_f = ct.astype(values, ct.float32)
    zero_f = ct.zeros((ROW_BLOCK, COL_BLOCK), dtype=ct.float32)
    col_mask_2d = ct.reshape(col_mask, (1, COL_BLOCK))
    values_masked = ct.where(col_mask_2d, values_f, zero_f)
    partial = ct.sum(values_masked, axis=0)
    ct.scatter(partial_ptr, (ct.full((COL_BLOCK,), row_tile, dtype=ct.int32), cols),
               partial, mask=col_mask)


@ct.kernel
def _col_finalize_bf16_kernel(
    partial_ptr,      # f32 [num_row_tiles, VOCAB]
    out_ptr,          # f32 [VOCAB]  (bf16-rounded then back to f32)
    NUM_ROW_TILES: ct.Constant[int],
    VOCAB_N: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
    ROW_TILE_BLOCK: ct.Constant[int],
):
    col_tile = ct.bid(0)
    cols = col_tile * COL_BLOCK + ct.arange(COL_BLOCK, dtype=ct.int32)
    col_mask = cols < VOCAB_N
    values = ct.load(partial_ptr, index=(0, col_tile),
                     shape=(ROW_TILE_BLOCK, COL_BLOCK),
                     padding_mode=ct.PaddingMode.ZERO)
    row_idx = ct.arange(ROW_TILE_BLOCK, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < NUM_ROW_TILES, (ROW_TILE_BLOCK, 1))
    zero_f = ct.zeros((ROW_TILE_BLOCK, COL_BLOCK), dtype=ct.float32)
    valid = ct.where(row_mask, values, zero_f)
    total = ct.sum(valid, axis=0)
    total_bf = ct.astype(total, ct.bfloat16)
    total_f = ct.astype(total_bf, ct.float32)
    ct.scatter(out_ptr, (cols,), total_f, mask=col_mask)


def _next_p2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


@oracle_impl(hardware="B200", point="3953bf5d")
def oracle_forward(inputs):
    (
        numerator, denominator, labels, logits, shift0, shift1, residual,
        _shape0, _shape1, _shape2, _shape3, _shape4, _shape5, _shape6, _shape7,
    ) = inputs
    device = logits.device
    rows = ROWS
    vocab = VOCAB
    vocab_pad = VOCAB_PAD

    labels_flat = labels.view(-1)
    scale = (numerator / denominator).view(())
    ne = labels_flat != -100
    safe_label = torch.where(ne, labels_flat, torch.zeros_like(labels_flat))
    in_vocab = (safe_label >= 0) & (safe_label < vocab)
    active = ne & in_vocab
    neg_scale_bf = (-scale).to(torch.bfloat16).to(torch.float32)
    row_sum_f = torch.where(active, neg_scale_bf.expand_as(safe_label),
                            torch.zeros_like(labels_flat, dtype=torch.float32))

    shift0_1d = shift0.view(rows).contiguous()
    shift1_1d = shift1.view(rows).contiguous()

    # logits comes in with stride (128104, 1) — [rows, vocab] view with padded stride
    # Reinterpret: full row is VOCAB_PAD bf16s where cols [VOCAB, VOCAB_PAD) are unused
    # Actually the shape is (rows, VOCAB) with stride (VOCAB_PAD, 1). So the underlying
    # memory is [rows, VOCAB_PAD]. We can rebuild the storage as a full [rows,VOCAB_PAD] view.
    logits_storage = torch.empty_strided(
        (rows, vocab_pad), (vocab_pad, 1),
        device=device, dtype=torch.bfloat16,
    )
    logits_storage[:, :vocab].copy_(logits.view(rows, vocab))
    logits_storage[:, vocab:].zero_()

    residual_2d = residual.view(rows, vocab).contiguous()

    # Outputs
    out_padded = torch.empty_strided(
        (rows, vocab_pad), (vocab_pad, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_trans = torch.empty_strided(
        (vocab_pad, rows), (rows, 1),
        device=device, dtype=torch.bfloat16,
    )

    BLOCK_N = 512
    n_col_blocks = (vocab_pad + BLOCK_N - 1) // BLOCK_N

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, n_col_blocks, 1), _xent_bwd_kernel,
        (logits_storage, residual_2d, row_sum_f, shift0_1d, shift1_1d, safe_label,
         out_padded, out_trans, rows, vocab, vocab_pad, BLOCK_N),
    )

    # Column sum kernel over out_padded (bf16 [rows, vocab_pad])
    ROW_BLOCK_COL = 128
    COL_BLOCK_COL = 128
    num_row_tiles = rows // ROW_BLOCK_COL
    num_col_tiles = (vocab + COL_BLOCK_COL - 1) // COL_BLOCK_COL
    partial = torch.zeros((num_row_tiles, vocab), device=device, dtype=torch.float32)
    ct.launch(
        stream, (num_row_tiles, num_col_tiles, 1), _column_sum_kernel,
        (out_padded, partial, ROW_BLOCK_COL, COL_BLOCK_COL, vocab, vocab_pad),
    )

    out_sum = torch.empty((vocab,), device=device, dtype=torch.float32)
    row_tile_block = _next_p2(num_row_tiles)
    ct.launch(
        stream, (num_col_tiles, 1, 1), _col_finalize_bf16_kernel,
        (partial, out_sum, num_row_tiles, vocab, COL_BLOCK_COL, row_tile_block),
    )

    return out_padded, out_trans, out_sum
