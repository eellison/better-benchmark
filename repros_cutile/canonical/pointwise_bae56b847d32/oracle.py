"""cuTile port of pointwise_bae56b847d32: BEiT relative-position-bias gather.

Loads the [732, HEADS] table via gather using indices from a [197, 197] i64
tensor and writes the [1, HEADS, 197, 197] output. Uses a padded per-head
buffer since ELEMENTS=197*197 is not a power of two.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 197
HEADS = 12
ELEMENTS = SEQ * SEQ  # 38809
OUT_SHAPE = (1, HEADS, SEQ, SEQ)
OUT_STRIDE = (HEADS * ELEMENTS, ELEMENTS, SEQ, 1)


@ct.kernel
def _relative_bias_perhead_kernel(
    index_ptr,     # i64 padded [PADDED_ELEM]
    table_ptr,     # bf16 [732 * HEADS]
    out_ptr,       # bf16 [HEADS, PADDED_ELEM]
    HEADS_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    head = ct.bid(0)
    spatial_block = ct.bid(1)

    lane = spatial_block * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    idx = ct.gather(index_ptr, lane)  # i64 tile
    table_off = idx * HEADS_C + head
    values = ct.gather(table_ptr, table_off)  # bf16 tile shape (BLOCK,)

    # Store as row (1, BLOCK) into 2D out at (head, spatial_block).
    ct.store(out_ptr, index=(head, spatial_block), tile=ct.reshape(values, (1, BLOCK)))


@oracle_impl(hardware="B200", point="86e210b7", BLOCK=128)
def oracle_forward(inputs, *, BLOCK: int):
    index, table, _shape_param_0 = inputs

    padded_elem = ((ELEMENTS + BLOCK - 1) // BLOCK) * BLOCK
    device = table.device

    padded_out = torch.empty(
        (HEADS, padded_elem), device=device, dtype=table.dtype
    )

    index_padded = torch.zeros(padded_elem, device=device, dtype=index.dtype)
    index_padded[:ELEMENTS] = index.contiguous().view(-1)

    table_flat = table.contiguous().view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (HEADS, padded_elem // BLOCK, 1),
        _relative_bias_perhead_kernel,
        (index_padded, table_flat, padded_out, HEADS, BLOCK),
    )

    out = padded_out[:, :ELEMENTS].contiguous().view(OUT_SHAPE)
    return out
