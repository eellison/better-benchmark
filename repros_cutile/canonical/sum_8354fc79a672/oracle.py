"""cuTile port of sum_8354fc79a672: OPT attention head permute/scale/sum.

The input is bf16[4, 12, 2048, 64]. Permute to [4, 2048, 12, 64] -> [8192, 768],
scale by 0.125 in f32 with bf16 rounding, and hidden-column-sum as f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
HEADS = 12
SEQ = 2048
HEAD_DIM = 64
ROWS = BATCH * SEQ  # 8192
HIDDEN = HEADS * HEAD_DIM  # 768
SCALE = 0.125


@ct.kernel
def _permute_scale_kernel(
    src,   # (BATCH, HEADS, SEQ, HEAD_DIM) bf16
    dst,   # (BATCH, SEQ, HEADS, HEAD_DIM) bf16 -- viewed as (ROWS, HIDDEN)
    BLOCK_S: ct.Constant[int],
):
    b = ct.bid(0)
    s_block = ct.bid(1)
    h = ct.bid(2)
    # Load (1, 1, BLOCK_S, HEAD_DIM) from src at (b, h, s_block, 0)
    v = ct.load(src, index=(b, h, s_block, 0), shape=(1, 1, BLOCK_S, HEAD_DIM))
    v_f = ct.astype(v, ct.float32) * SCALE
    v_bf = ct.astype(v_f, ct.bfloat16)
    # Reshape to (1, BLOCK_S, 1, HEAD_DIM) for store into (BATCH, SEQ, HEADS, HEAD_DIM)
    v_bf_4d = ct.reshape(v_bf, (1, BLOCK_S, 1, HEAD_DIM))
    ct.store(dst, index=(b, s_block, h, 0), tile=v_bf_4d)


@ct.kernel
def _partial_sum_kernel(
    scaled_flat,  # (ROWS * HIDDEN,) bf16 (contiguous view of scaled)
    partial,      # (num_row_blocks, HIDDEN) f32
    BLOCK_ROWS: ct.Constant[int],
    NUM_HIDDEN_BLOCKS: ct.Constant[int],
    BLOCK_HIDDEN: ct.Constant[int],
):
    row_block = ct.bid(0)
    hidden_block = ct.bid(1)
    # Load (BLOCK_ROWS, BLOCK_HIDDEN) from scaled treated as (ROWS, HIDDEN)
    # We use raw gather: offsets = (row_block*BLOCK_ROWS + r)*HIDDEN + hidden_block*BLOCK_HIDDEN + c
    rows_local = ct.arange(BLOCK_ROWS, dtype=ct.int32)
    cols_local = ct.arange(BLOCK_HIDDEN, dtype=ct.int32)
    row_ids = row_block * BLOCK_ROWS + rows_local
    hidden_ids = hidden_block * BLOCK_HIDDEN + cols_local

    row_2d = ct.reshape(row_ids, (BLOCK_ROWS, 1))
    hid_2d = ct.reshape(hidden_ids, (1, BLOCK_HIDDEN))
    offsets = row_2d * HIDDEN + hid_2d
    v = ct.gather(scaled_flat, offsets, padding_value=0.0)
    v_f = ct.astype(v, ct.float32)
    part_sum = ct.sum(v_f, axis=0)  # (BLOCK_HIDDEN,)
    # Store into partial at row_block, hidden_ids
    part_off = row_block * HIDDEN + hidden_ids
    ct.scatter(partial, part_off, part_sum)


@ct.kernel
def _final_sum_kernel(
    partial_flat,  # (num_row_blocks * HIDDEN,) f32
    reduced,       # (HIDDEN,) f32
    NUM_TILES: ct.Constant[int],  # padded power-of-2
    NUM_ROW_BLOCKS: ct.Constant[int],
    BLOCK_HIDDEN: ct.Constant[int],
):
    hidden_block = ct.bid(0)
    hidden_ids = hidden_block * BLOCK_HIDDEN + ct.arange(BLOCK_HIDDEN, dtype=ct.int32)
    tiles = ct.arange(NUM_TILES, dtype=ct.int32)
    tile_mask = tiles < NUM_ROW_BLOCKS
    tile_2d = ct.reshape(tiles, (NUM_TILES, 1))
    hid_2d = ct.reshape(hidden_ids, (1, BLOCK_HIDDEN))
    valid = ct.reshape(tile_mask, (NUM_TILES, 1))
    offsets = tile_2d * HIDDEN + hid_2d
    v = ct.gather(partial_flat, offsets, mask=valid, padding_value=0.0)
    summed = ct.sum(v, axis=0)
    rounded = ct.astype(ct.astype(summed, ct.bfloat16), ct.float32)
    ct.scatter(reduced, hidden_ids, rounded)


@oracle_impl(hardware="B200", point="f696cede", BLOCK_ROWS=128, BLOCK_HIDDEN=64)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_HIDDEN):
    x, _shape0, _shape1, _shape2 = inputs
    scaled = torch.empty_strided(
        (BATCH, SEQ, HEADS, HEAD_DIM),
        (SEQ * HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEAD_DIM, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    def next_pow2(v):
        p = 1
        while p < v:
            p *= 2
        return p

    stream = torch.cuda.current_stream()

    BLOCK_S = 64  # covers 2048 in 32 blocks; must be pow-of-2 <= SEQ
    ct.launch(
        stream,
        (BATCH, SEQ // BLOCK_S, HEADS),
        _permute_scale_kernel,
        (x, scaled, BLOCK_S),
    )

    num_row_blocks = (ROWS + BLOCK_ROWS - 1) // BLOCK_ROWS
    partial = torch.empty((num_row_blocks, HIDDEN), device=x.device, dtype=torch.float32)
    scaled_flat = scaled.view(ROWS, HIDDEN).reshape(-1)
    ct.launch(
        stream,
        (num_row_blocks, HIDDEN // BLOCK_HIDDEN, 1),
        _partial_sum_kernel,
        (scaled_flat, partial.reshape(-1), BLOCK_ROWS, HIDDEN // BLOCK_HIDDEN, BLOCK_HIDDEN),
    )

    reduced = torch.empty((HIDDEN,), device=x.device, dtype=torch.float32)
    ct.launch(
        stream,
        (HIDDEN // BLOCK_HIDDEN, 1, 1),
        _final_sum_kernel,
        (
            partial.reshape(-1),
            reduced,
            next_pow2(num_row_blocks),
            num_row_blocks,
            BLOCK_HIDDEN,
        ),
    )

    scaled_2d = scaled.view(ROWS, HIDDEN)
    return scaled_2d, scaled_2d.permute(1, 0), reduced
