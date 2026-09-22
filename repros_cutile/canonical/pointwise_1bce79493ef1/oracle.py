"""cuTile port of pointwise_1bce79493ef1: MobileBERT cat + right-pad.

Concatenates permute(arg0) (top: [128, 30522]) with arg1 (bottom: [384, 30522])
producing [512, 30522], then right-pads to [512, 30528].

Strategy:
  1. Zero-initialize the output tensor with torch (handles pad + easy).
  2. cuTile kernel copies the interior [512, 30522] region. Each output row
     comes from either the top or the bottom source. Non-power-of-2 column
     count (30522) is handled by using ct.scatter with a column mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


TOP_ROWS = 128
BOTTOM_ROWS = 384
OUT_ROWS = TOP_ROWS + BOTTOM_ROWS  # 512
VOCAB = 30522
PADDED_VOCAB = 30528
OUT_SHAPE = (OUT_ROWS, PADDED_VOCAB)
OUT_STRIDE = (PADDED_VOCAB, 1)


@ct.kernel
def _cat_kernel(
    top_flat_ptr,     # bf16 [30522 * 128] flat (permute output)
    bottom_flat_ptr,  # bf16 [384 * 30522] flat
    out_flat_ptr,     # bf16 [512 * 30528] flat
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    TOP_ROWS_: ct.Constant[int],
    VOCAB_: ct.Constant[int],
    PADDED_VOCAB_: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    rows = row_block * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    cols = col_block * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)

    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))

    in_cat = cols_2d < VOCAB_
    from_top = rows_2d < TOP_ROWS_

    safe_col = ct.where(
        in_cat,
        cols_2d,
        ct.zeros((1, BLOCK_N), dtype=ct.int32),
    )
    # top layout: original arg0 is [30522, 128]. permute(1,0) has shape
    # [128, 30522] but is strided (stride = (1, 128)). The flat view of the
    # underlying storage means top_flat[c*128 + r] for row r, col c.
    safe_row_top = ct.where(
        from_top,
        rows_2d,
        ct.zeros((BLOCK_M, 1), dtype=ct.int32),
    )
    top_offsets = safe_col * TOP_ROWS_ + safe_row_top
    top_values = ct.gather(top_flat_ptr, top_offsets)

    safe_row_bot = ct.where(
        from_top,
        ct.zeros((BLOCK_M, 1), dtype=ct.int32),
        rows_2d - TOP_ROWS_,
    )
    bot_offsets = safe_row_bot * VOCAB_ + safe_col
    bot_values = ct.gather(bottom_flat_ptr, bot_offsets)

    zero = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    values = ct.where(from_top, top_values, bot_values)
    values = ct.where(in_cat, values, zero)

    out_offsets = rows_2d * PADDED_VOCAB_ + cols_2d
    valid = (rows_2d < 512) & (cols_2d < PADDED_VOCAB_)
    ct.scatter(out_flat_ptr, out_offsets, values, mask=valid)


@oracle_impl(
    hardware="B200",
    point="0fc085ac",
    BLOCK_M=8,
    BLOCK_N=256,
)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    top, bottom, _pad = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=top.device,
        dtype=torch.bfloat16,
    )
    # top is [30522, 128] contiguous. Its .permute(1, 0) is [128, 30522] with
    # strides (1, 128). We index the underlying storage directly.
    top_flat = top.view(-1)
    bottom_flat = bottom.contiguous().view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((OUT_ROWS + BLOCK_M - 1) // BLOCK_M,
         (PADDED_VOCAB + BLOCK_N - 1) // BLOCK_N,
         1),
        _cat_kernel,
        (top_flat, bottom_flat, out_flat, BLOCK_M, BLOCK_N,
         TOP_ROWS, VOCAB, PADDED_VOCAB),
    )
    return out
