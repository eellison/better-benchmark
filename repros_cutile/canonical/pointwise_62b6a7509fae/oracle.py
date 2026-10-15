"""cuTile port of pointwise_62b6a7509fae: BEiT relative-position-bias gather.

For each (head, row, col) in [12, 197, 197]:
  idx = index[row, col]  (i64)
  base[0, head, row, col] = bfloat16(table[idx, head])
For col in [197, 200): base[0, head, row, col] = 0.
Expanded [128, 12, 197, 197] is a strided view of base.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 197
HEADS = 12
PADDED_SEQ = 200
BATCH = 128
PAD_COLS = PADDED_SEQ - SEQ
VISIBLE_ELEMENTS = HEADS * SEQ * SEQ  # 465468
PAD_ELEMENTS = HEADS * SEQ * PAD_COLS  # 7092
BASE_SHAPE = (1, HEADS, SEQ, PADDED_SEQ)
BASE_STRIDE = (HEADS * SEQ * PADDED_SEQ, SEQ * PADDED_SEQ, PADDED_SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)
EXPAND_STRIDE = (0, SEQ * PADDED_SEQ, PADDED_SEQ, 1)


@ct.kernel
def _relative_bias_kernel(
    index_flat_ptr,   # i64 (SEQ*SEQ,) view of index
    table_flat_ptr,   # f32 (732 * HEADS,) flat view
    out_flat_ptr,     # bf16 (HEADS*SEQ*PADDED_SEQ,)
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    off = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    mask = off < TOTAL

    hw = SEQ * SEQ
    head = off // hw
    pos = off - head * hw
    row = pos // SEQ
    col = pos - row * SEQ

    # index[row, col]
    idx = ct.gather(index_flat_ptr, row * SEQ + col, mask=mask, padding_value=0)
    idx_i32 = ct.astype(idx, ct.int32)
    # Table is [732, 12]. table[idx, head] flat offset = idx * HEADS + head.
    values = ct.gather(table_flat_ptr, idx_i32 * HEADS + head, mask=mask, padding_value=0.0)
    values_bf = ct.astype(values, ct.bfloat16)
    out_offsets = head * (SEQ * PADDED_SEQ) + row * PADDED_SEQ + col
    ct.scatter(out_flat_ptr, out_offsets, values_bf, mask=mask)


@ct.kernel
def _pad_zero_kernel(
    out_flat_ptr,     # bf16
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    off = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    mask = off < TOTAL

    row_head = off // PAD_COLS
    pad_col = SEQ + (off - row_head * PAD_COLS)
    pad_row = row_head - (row_head // SEQ) * SEQ
    pad_head = row_head // SEQ
    out_off = pad_head * (SEQ * PADDED_SEQ) + pad_row * PADDED_SEQ + pad_col
    zero = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    ct.scatter(out_flat_ptr, out_off, zero, mask=mask)


@oracle_impl(hardware="B200", point="0b512413", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    index, table, *_shape_params = inputs
    base = torch.empty_strided(BASE_SHAPE, BASE_STRIDE,
                               device=table.device, dtype=torch.bfloat16)
    index_flat = index.reshape(-1)
    base_flat = base.view(-1)
    table_flat = table.reshape(-1)

    stream = torch.cuda.current_stream()
    # Emit values
    total_v = VISIBLE_ELEMENTS
    ct.launch(
        stream,
        ((total_v + BLOCK - 1) // BLOCK, 1, 1),
        _relative_bias_kernel,
        (index_flat, table_flat, base_flat, total_v, BLOCK),
    )
    # Emit pad zeros
    total_p = PAD_ELEMENTS
    ct.launch(
        stream,
        ((total_p + BLOCK - 1) // BLOCK, 1, 1),
        _pad_zero_kernel,
        (base_flat, total_p, BLOCK),
    )
    expanded = torch.as_strided(base, EXPAND_SHAPE, EXPAND_STRIDE)
    return base, expanded
