"""cuTile port of pointwise_260db4f7087d: XLNet slice_scatter+view layout materialization.

Writes the final [256, 512, 1024] contiguous bf16 output row-by-row: for each row,
copy the corresponding arg1_1 slice (512 elements) into the prefix, and the
arg0_1 flattened slice (511 x 1024 + 512 = 523776 elements) into the tail.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 256
OUT_ROW_ELEMS = 512 * 1024  # 524288
SRC_ROW_ELEMS = 512 * 1023   # 523776
BLOCK = 512


@ct.kernel
def _copy_tail_kernel(
    src_ptr,   # (ROWS, SRC_ROW_ELEMS)
    dst_ptr,   # (ROWS, OUT_ROW_ELEMS)
    OFFSET: ct.Constant[int],
    BLOCK_: ct.Constant[int],
):
    row = ct.bid(0)
    j_block = ct.bid(1)
    values = ct.load(src_ptr, index=(row, j_block), shape=(1, BLOCK_))
    # Store: destination column index (in units of BLOCK_) = j_block + OFFSET/BLOCK_
    ct.store(dst_ptr, index=(row, j_block + OFFSET // BLOCK_), tile=values)


@ct.kernel
def _copy_prefix_kernel(
    base_ptr,  # (ROWS, 512) — but stored as arg1_1 whole tile view of first row
    dst_ptr,   # (ROWS, OUT_ROW_ELEMS)
    BLOCK_: ct.Constant[int],
):
    row = ct.bid(0)
    values = ct.load(base_ptr, index=(row, 0), shape=(1, BLOCK_))
    ct.store(dst_ptr, index=(row, 0), tile=values)


@oracle_impl(hardware="B200", point="0578bbc7")
def oracle_forward(inputs):
    arg0_1, arg1_1, _shape0, _shape1, _shape2, _shape3 = inputs
    # arg0_1: [16, 16, 512, 1023] bf16, contiguous — flat [ROWS, SRC_ROW_ELEMS]
    # arg1_1: [16, 16, 1024, 512] bf16, contiguous — flat [ROWS, OUT_ROW_ELEMS]
    # The prefix comes from arg1_1[:,:,0,:] — first 512 elems of each (b,h) row.
    src = arg0_1.reshape(ROWS, SRC_ROW_ELEMS)
    base = arg1_1.reshape(ROWS, OUT_ROW_ELEMS)

    out = torch.empty_strided(
        (ROWS, 512, 1024),
        (OUT_ROW_ELEMS, 1024, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    dst = out.view(ROWS, OUT_ROW_ELEMS)

    stream = torch.cuda.current_stream()
    # Tail: SRC_ROW_ELEMS / BLOCK = 1023 tiles per row.
    ct.launch(
        stream,
        (ROWS, SRC_ROW_ELEMS // BLOCK, 1),
        _copy_tail_kernel,
        (src, dst, 512, BLOCK),
    )
    # Prefix: copy first BLOCK=512 elems from base to dst.
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _copy_prefix_kernel,
        (base, dst, BLOCK),
    )
    return out
