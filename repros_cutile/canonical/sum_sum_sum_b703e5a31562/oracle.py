"""cuTile port of sum_sum_sum_b703e5a31562: ALBERT 12x column sums + head layout materialization."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 4096
CHANNELS = 4096
SCALE = 0.3535533905932738


@ct.kernel
def _column_sum_kernel(
    mat_ptr,   # (ROWS, CHANNELS) bf16
    out_ptr,   # (num_row_blocks, CHANNELS) f32 partial store
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_tile = ct.bid(0)
    row_tile = ct.bid(1)

    vals = ct.load(mat_ptr, index=(row_tile, col_tile), shape=(BLOCK_ROWS, BLOCK_COLS))
    vals_f = ct.astype(vals, ct.float32)
    partial = ct.sum(vals_f, axis=0, keepdims=True)
    ct.store(out_ptr, index=(row_tile, col_tile), tile=partial)


@ct.kernel
def _layout_and_sum_kernel(
    src_ptr,  # (512, 512, 64) bf16 = (8, 64, 512, 64) view
    clone_out_ptr,  # (ROWS, CHANNELS) = (8, 512, 64, 64) bf16
    partial_ptr,  # (num_row_blocks, CHANNELS) f32 partial store
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_tile = ct.bid(0)
    row_tile = ct.bid(1)
    # src is (8, 64, 512, 64), clone is (8, 512, 64, 64)
    # ROWS = 8 * 512, CHANNELS = 64 * 64
    # row_tile picks BLOCK_ROWS rows. If BLOCK_ROWS=128, each tile covers rows in a range.
    # Each row in the clone corresponds to (batch, seq) = (row // 512, row % 512).
    # Each col corresponds to (head, dim) = (col // 64, col % 64).
    # Values = src[batch, head, seq, dim] * SCALE.
    # But we can't easily tile 2D over these interleaved dims without careful reshape.
    #
    # Simplest approach: load a strided tile from src via 4D indexing.
    # The clone tile at (row_tile, col_tile) covers:
    #   rows [row_tile * BLOCK_ROWS, ... + BLOCK_ROWS) => batches, seqs
    #   cols [col_tile * BLOCK_COLS, ... + BLOCK_COLS) => heads, dims
    # For BLOCK_ROWS=128 and CHANNELS=4096, 128 rows cover partial batches or full sequences (128 < 512 seq per batch)
    # For BLOCK_COLS=64 = one head's dim size, col_tile picks a single head. So the tile is a
    # rectangular chunk within one (batch, head) block.
    # Given row_tile * BLOCK_ROWS = r0, batch = r0 // 512, seq_start = r0 % 512.
    # In principle we can just load from src at strided coords, but easier: reshape ala Repro.
    # Instead we materialize by reading from src via strides. However cuTile doesn't allow scalar addressing.
    #
    # Since we've viewed src as (8, 64, 512, 64) and clone as (8, 512, 64, 64), we can process
    # per-head chunks: col_tile determines the head, row_tile determines the (batch,seq)-range.
    # For BLOCK_COLS=64, col_tile = head. For BLOCK_ROWS=128, we cover 128 rows.
    # But rows span batches when they cross the 512 boundary. Let's assert BLOCK_ROWS divides 512.
    #
    # If row_tile * BLOCK_ROWS lies fully within one batch:
    #   batch = row_tile * BLOCK_ROWS // 512
    #   seq_offset_in_tile = (row_tile * BLOCK_ROWS) % 512
    # We'll enforce BLOCK_ROWS divides 512.
    #
    # Load src[batch, head, seq_offset:seq_offset+BLOCK_ROWS, 0:BLOCK_COLS]
    seq_per_batch = 512
    rows_per_batch_block = seq_per_batch // BLOCK_ROWS  # 512//128 = 4
    batch = row_tile // rows_per_batch_block
    seq_tile_in_batch = row_tile - batch * rows_per_batch_block
    head = col_tile

    # Load (1, 1, BLOCK_ROWS, BLOCK_COLS) then squeeze
    vals = ct.load(src_ptr, index=(batch, head, seq_tile_in_batch, 0),
                   shape=(1, 1, BLOCK_ROWS, BLOCK_COLS))
    vals_2d = ct.reshape(vals, (BLOCK_ROWS, BLOCK_COLS))
    scaled = ct.astype(ct.astype(vals_2d, ct.float32) * SCALE, ct.bfloat16)

    # Store to clone at (batch, seq_tile_in_batch, head, 0). The clone is (8, 512, 64, 64) but
    # we've viewed the flat storage as (ROWS, CHANNELS) = (4096, 4096). The layout of clone:
    #   clone_4d[batch, seq, head, dim] is at flat position (batch*512 + seq)*4096 + head*64 + dim
    # which matches (row=batch*512+seq, col=head*64+dim) in the (4096, 4096) view.
    # But we can't easily index 4D directly if we pass the 2D view.
    # Let's pass the 4D view of clone.
    scaled_4d = ct.reshape(scaled, (1, BLOCK_ROWS, 1, BLOCK_COLS))
    ct.store(clone_out_ptr, index=(batch, seq_tile_in_batch, head, 0), tile=scaled_4d)

    partial = ct.sum(ct.astype(scaled, ct.float32), axis=0, keepdims=True)
    ct.store(partial_ptr, index=(row_tile, col_tile), tile=partial)


@ct.kernel
def _finalize_kernel(
    partials_ptr,  # (12, num_row_blocks, CHANNELS) f32
    out_ptr,       # (CHANNELS,) f32
    NUM_ROW_BLOCKS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_tile = ct.bid(0)
    # Load and sum for each matrix
    # partials_ptr has shape (12, NUM_ROW_BLOCKS, CHANNELS)
    # For each of the 12 partials, we need to sum across NUM_ROW_BLOCKS and add
    # But we cannot conditionally sum in cuTile easily with just BLOCK_COLS tile. Since
    # NUM_ROW_BLOCKS is a compile-time constant, we can unroll.
    total = ct.zeros(shape=(BLOCK_COLS,), dtype=ct.float32)
    for i in range(12):
        # Load the (NUM_ROW_BLOCKS, BLOCK_COLS) partials for matrix i
        block_partials = ct.load(partials_ptr, index=(i, 0, col_tile),
                                 shape=(1, NUM_ROW_BLOCKS, BLOCK_COLS))
        block_2d = ct.reshape(block_partials, (NUM_ROW_BLOCKS, BLOCK_COLS))
        sub = ct.sum(block_2d, axis=0)
        # Round through bf16 then back to f32 (matches Repro numerics)
        rounded = ct.astype(ct.astype(sub, ct.bfloat16), ct.float32)
        total = total + rounded
    ct.store(out_ptr, index=(col_tile,), tile=total)


@oracle_impl(
    hardware="B200",
    point="4a97732f",
    BLOCK_ROWS=128,
    BLOCK_COLS=64,
    FINAL_BLOCK_COLS=16,
)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_COLS, FINAL_BLOCK_COLS):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
        *_shape_args,
    ) = inputs

    num_row_blocks = ROWS // BLOCK_ROWS
    device = arg0.device

    clone = torch.empty_strided(
        (ROWS, CHANNELS),
        (CHANNELS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (12, num_row_blocks, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    # View arg11 as (8, 64, 512, 64) and clone as (8, 512, 64, 64)
    src_4d = arg11.view(8, 64, 512, 64)
    clone_4d = clone.view(8, 512, 64, 64)

    stream = torch.cuda.current_stream()

    # 11 column sums for arg0..arg10
    grid = (CHANNELS // BLOCK_COLS, num_row_blocks, 1)
    for i, mat in enumerate([arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10]):
        ct.launch(
            stream,
            grid,
            _column_sum_kernel,
            (mat, partials[i], BLOCK_ROWS, BLOCK_COLS),
        )
    # Layout + column sum for arg11
    ct.launch(
        stream,
        grid,
        _layout_and_sum_kernel,
        (src_4d, clone_4d, partials[11], BLOCK_ROWS, BLOCK_COLS),
    )

    # Finalize: sum partials + bf16 rounding roundtrip + accumulate
    ct.launch(
        stream,
        (CHANNELS // FINAL_BLOCK_COLS, 1, 1),
        _finalize_kernel,
        (partials, out, num_row_blocks, FINAL_BLOCK_COLS),
    )

    return clone, clone.permute(1, 0), out
