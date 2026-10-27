"""cuTile port of pointwise_de8967598ec8: segment-aware causal 0/-inf mask, materialized 12x."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
N_ELEMENTS = BATCH * SEQ * SEQ
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@ct.kernel
def _segment_causal_mask_12_kernel(
    seg_ptr,  # (BATCH, SEQ) i64/i32
    out0, out1, out2, out3, out4, out5,
    out6, out7, out8, out9, out10, out11,
    SEQ_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    batch = ct.bid(0)
    row = ct.bid(1)
    col_block = ct.bid(2)

    row_seg_tile = ct.load(seg_ptr, index=(batch, row), shape=(1, 1))
    col_seg = ct.load(seg_ptr, index=(batch, col_block), shape=(1, BLOCK_N))
    # Column indices in tile.
    col_ids = ct.arange(BLOCK_N, dtype=ct.int32) + col_block * BLOCK_N
    col_ids = ct.reshape(col_ids, (1, BLOCK_N))
    row_id_full = ct.full(shape=(1, BLOCK_N), fill_value=row, dtype=ct.int32)

    keep_causal = col_ids <= row_id_full
    keep_seg = col_seg == row_seg_tile  # broadcast row_seg (1,1) to (1, BLOCK_N)
    keep = keep_causal & keep_seg

    zero = ct.full(shape=(1, BLOCK_N), fill_value=0.0, dtype=ct.float32)
    ninf = ct.full(shape=(1, BLOCK_N), fill_value=float("-inf"), dtype=ct.float32)
    value = ct.where(keep, zero, ninf)
    value_bf = ct.astype(value, ct.bfloat16)

    # Store to all 12 outputs. Each output is shape (BATCH, 1, SEQ, SEQ);
    # index into (b, 0, row, col_block).
    ct.store(out0, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out1, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out2, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out3, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out4, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out5, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out6, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out7, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out8, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out9, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out10, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))
    ct.store(out11, index=(batch, 0, row, col_block), tile=ct.reshape(value_bf, (1, 1, 1, BLOCK_N)))


@oracle_impl(hardware="B200", point="0105f520", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    segments, _shape0 = inputs
    outputs = [
        torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=segments.device,
            dtype=torch.bfloat16,
        )
        for _ in range(12)
    ]
    seg_2d = segments.view(BATCH, SEQ)
    stream = torch.cuda.current_stream()
    grid = (BATCH, SEQ, SEQ // BLOCK_N)
    ct.launch(
        stream,
        grid,
        _segment_causal_mask_12_kernel,
        (seg_2d, *outputs, SEQ, BLOCK_N),
    )
    return tuple(outputs)
