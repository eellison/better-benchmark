"""cuTile port of sum_ed1b436345f1: DeBERTa div + head-reorder + column-sum.

Two cuTile kernels:
  1. Layout-div-reduce partials: for each (ROW_BLOCK x COL_BLOCK) tile of the
     output [4096, 1536] tensor, compute the divided bf16 values (with the
     head-reorder index mapping) and store both the tile and per-row-block
     fp32 column partials.
  2. Final column sum: for each output column, sum the row-block partials and
     round through bf16 to fp32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
HEADS = 24
QUERY = 64
KEY = 512
ROWS = BATCH * KEY       # 4096
FEATURES = HEADS * QUERY # 1536
ROW_BLOCK = 64
COL_BLOCK = 64
NUM_ROW_BLOCKS = ROWS // ROW_BLOCK  # 64


@ct.kernel
def _layout_div_reduce_kernel(
    x_ptr,          # bf16 [192*64*512] (flat)
    scale_ptr,      # bf16 [1] (view of the scalar)
    out_ptr,        # bf16 [ROWS, FEATURES]
    partial_ptr,    # f32 [NUM_ROW_BLOCKS, FEATURES]
    ROW_BLOCK_: ct.Constant[int],
    COL_BLOCK_: ct.Constant[int],
    HEADS_: ct.Constant[int],
    QUERY_: ct.Constant[int],
    KEY_: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    rows = row_block * ROW_BLOCK_ + ct.arange(ROW_BLOCK_, dtype=ct.int32)
    cols = col_block * COL_BLOCK_ + ct.arange(COL_BLOCK_, dtype=ct.int32)

    batch = rows // KEY_
    key = rows - batch * KEY_
    head = cols // QUERY_
    query = cols - head * QUERY_
    batch_head = ct.reshape(batch, (ROW_BLOCK_, 1)) * HEADS_ + ct.reshape(head, (1, COL_BLOCK_))
    query_2d = ct.reshape(query, (1, COL_BLOCK_))
    key_2d = ct.reshape(key, (ROW_BLOCK_, 1))

    # x has strides based on (192, 64, 512) contiguous:
    #   stride_bh = 64*512 = 32768
    #   stride_q  = 512
    #   stride_k  = 1
    x_offsets = batch_head * (QUERY_ * KEY_) + query_2d * KEY_ + key_2d

    scale = ct.load(scale_ptr, index=(0,), shape=(1,))
    scale_f = ct.astype(scale, ct.float32)
    scale_scalar = ct.reshape(scale_f, (1, 1))

    values = ct.gather(x_ptr, x_offsets)
    values_f = ct.astype(values, ct.float32)
    div_bf16 = ct.astype(values_f / scale_scalar, ct.bfloat16)

    ct.store(out_ptr, index=(row_block, col_block), tile=div_bf16)

    div_f32 = ct.astype(div_bf16, ct.float32)
    partial = ct.sum(div_f32, axis=0)  # (COL_BLOCK,)
    partial_2d = ct.reshape(partial, (1, COL_BLOCK_))
    ct.store(partial_ptr, index=(row_block, col_block), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,    # f32 [NUM_ROW_BLOCKS, FEATURES]
    out_ptr,        # f32 [FEATURES]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    tile = ct.load(
        partial_ptr,
        index=(0, col_block),
        shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(tile, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(col_block,), tile=rounded)


@oracle_impl(hardware="B200", point="9929f4a9")
def oracle_forward(inputs):
    x, scale, _shape0, _shape1, shape2, _shape3 = inputs
    # x is (192, 64, 512) contiguous; we need a flat 1D view for gather.
    x_flat = x.contiguous().view(-1)
    scale_1d = scale.view(1)

    out_base = torch.empty(
        tuple(int(dim) for dim in shape2),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (NUM_ROW_BLOCKS, FEATURES),
        device=x.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty((FEATURES,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUM_ROW_BLOCKS, FEATURES // COL_BLOCK, 1),
        _layout_div_reduce_kernel,
        (x_flat, scale_1d, out_base, partials, ROW_BLOCK, COL_BLOCK, HEADS, QUERY, KEY),
    )
    block_r = 1
    while block_r < NUM_ROW_BLOCKS:
        block_r *= 2
    ct.launch(
        stream,
        (FEATURES // COL_BLOCK, 1, 1),
        _final_sum_kernel,
        (partials, out_sum, block_r, COL_BLOCK),
    )
    out_transpose = torch.as_strided(out_base, (FEATURES, ROWS), (1, FEATURES))
    return out_base, out_transpose, out_sum
