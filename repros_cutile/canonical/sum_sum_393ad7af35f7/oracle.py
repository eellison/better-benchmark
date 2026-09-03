"""cuTile port of sum_sum_393ad7af35f7: MobileBERT masked-CE softmax-gradient.

Mirrors Triton's 2-kernel structure:
  - `_materialize_partials_kernel`: materialize padded output and emit per-row-block
    column partials via `ct.sum(..., axis=0)`.
  - `_finalize_sum_kernel`: reduce partials across row-blocks via `ct.sum`.

BLOCK_R=64, BLOCK_C=32 match Triton verbatim.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
VOCAB = 30522
PADDED_VOCAB = 30528


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _materialize_partials_kernel(
    numerator_ptr,     # f32 (1,)
    denominator_ptr,   # f32 (1,)
    labels_ptr,        # i64 (ROWS,)
    logits_ptr,        # bf16 (ROWS, VOCAB) — arg3 view with stride (PADDED_VOCAB, 1)
    residual_ptr,      # bf16 (ROWS, VOCAB) — arg4 view
    padded_out_ptr,    # bf16 (ROWS, PADDED_VOCAB)
    partial_ptr,       # f32 (num_row_blocks, PARTIAL_COLS)
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    num_tile = ct.load(numerator_ptr, index=(0,), shape=(1,))
    den_tile = ct.load(denominator_ptr, index=(0,), shape=(1,))
    scale = ct.astype(num_tile, ct.float32) / ct.astype(den_tile, ct.float32)

    labels = ct.load(labels_ptr, index=(row_block,), shape=(BLOCK_R,))
    active_label = labels != -100
    safe_label = ct.where(active_label, labels, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB)
    valid = active_label & in_vocab

    scale_bcast = ct.broadcast_to(scale, (BLOCK_R,))
    neg_scale = -scale_bcast
    # bf16 roundtrip
    row_sum_bf = ct.astype(neg_scale, ct.bfloat16)
    row_sum = ct.where(valid, ct.astype(row_sum_bf, ct.float32), 0.0)

    # Column indices for this tile
    cols = col_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    cols_2d = ct.reshape(cols, (1, BLOCK_C))
    vocab_col_mask = cols_2d < VOCAB
    safe_label_2d = ct.reshape(safe_label, (BLOCK_R, 1))
    valid_2d = ct.reshape(valid, (BLOCK_R, 1))
    row_sum_2d = ct.reshape(row_sum, (BLOCK_R, 1))

    # Load logits (arg3 view of shape (ROWS, VOCAB)). Right-edge tile has OOB
    # cols beyond VOCAB — pad with zero.
    logits = ct.load(logits_ptr, index=(row_block, col_block),
                     shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    logits_f = ct.astype(logits, ct.float32)
    exp_value = ct.exp(logits_f)

    match = (cols_2d == safe_label_2d) & valid_2d
    sparse = ct.where(match, row_sum_2d, 0.0)
    dense_delta = sparse - exp_value * row_sum_2d
    delta_bf16 = ct.astype(dense_delta, ct.bfloat16)

    # Load residual (arg4 view (ROWS, VOCAB)).
    residual = ct.load(residual_ptr, index=(row_block, col_block),
                       shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)

    out_value_f = ct.astype(residual, ct.float32) + ct.astype(delta_bf16, ct.float32)
    out_value = ct.astype(out_value_f, ct.bfloat16)

    # Padded output: zero OOB cols (>= VOCAB).
    zero_bf16 = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.bfloat16)
    padded_value = ct.where(vocab_col_mask, out_value, zero_bf16)
    ct.store(padded_out_ptr, index=(row_block, col_block), tile=padded_value)

    # Partial column sum of the padded output over rows.
    out_f = ct.astype(padded_value, ct.float32)
    partial = ct.sum(out_f, axis=0)  # (BLOCK_C,)
    ct.store(partial_ptr, index=(row_block, col_block),
             tile=ct.reshape(partial, (1, BLOCK_C)))


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr,       # f32 (NUM_ROW_BLOCKS, PARTIAL_COLS)
    sum_out_ptr,       # f32 (PARTIAL_COLS,)
    NUM_ROW_BLOCKS: ct.Constant[int],
    BLOCK_RB: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, col_block),
                     shape=(BLOCK_RB, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    row_idx = ct.arange(BLOCK_RB, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < NUM_ROW_BLOCKS, (BLOCK_RB, 1))
    valid = ct.where(row_mask, values, 0.0)
    reduced = ct.sum(valid, axis=0)
    ct.store(sum_out_ptr, index=(col_block,), tile=reduced)


@oracle_impl(hardware="B200", point="57ba714d")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
     shape0, shape1, shape2, shape3, shape4, shape5) = inputs
    device = arg3_1.device

    BLOCK_R = 64
    BLOCK_C = 32
    num_row_blocks = ROWS // BLOCK_R  # 512
    num_col_blocks = (VOCAB + BLOCK_C - 1) // BLOCK_C  # 954
    partial_cols = num_col_blocks * BLOCK_C  # 30528

    # Views
    labels_1d = arg2_1.view(-1).contiguous()  # (32768,)
    # arg3_1 is bf16[32768, 30522] with stride [30528, 1] — keep as-is; cuTile
    # partitions the (ROWS, VOCAB) space with padding for OOB tiles.
    arg3_view = arg3_1.view(ROWS, VOCAB) if arg3_1.dim() > 2 else arg3_1
    # arg4_1 is bf16[256,128,30522] contiguous — reshape to (ROWS, VOCAB).
    arg4_2d = arg4_1.view(ROWS, VOCAB)

    padded_out = torch.empty(ROWS, PADDED_VOCAB, device=device, dtype=torch.bfloat16)
    partial = torch.empty(num_row_blocks, partial_cols, device=device, dtype=torch.float32)
    sum_out = torch.empty(partial_cols, device=device, dtype=torch.float32)

    num_scalar = arg0_1.view(1)
    den_scalar = arg1_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_blocks, num_col_blocks, 1),
        _materialize_partials_kernel,
        (num_scalar, den_scalar, labels_1d,
         arg3_view, arg4_2d, padded_out, partial,
         BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream, (num_col_blocks, 1, 1),
        _finalize_sum_kernel,
        (partial, sum_out, num_row_blocks,
         _next_pow2(num_row_blocks), BLOCK_C),
    )

    # sum_out has partial_cols=30528 entries; return [:VOCAB]=30522.
    view_3 = sum_out[:VOCAB]
    # padded_out is (ROWS, PADDED_VOCAB) — matches the constant_pad_nd return shape.
    return view_3, padded_out
