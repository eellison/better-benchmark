"""cuTile port of pointwise_86c81d3b330e: masked row scatter-add (SCATTER_REDUCE).

Zeros a dense f32[out_rows, hidden] output, then atomically adds each source
row into the destination row indicated by the flattened index. Invalid
sources (row < 0, row >= out_rows, row == -1) are skipped via check_bounds
plus explicit index remap.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_scatter_add_kernel(
    index_arr,     # (total_sources,) i64
    values,        # (total_sources, hidden) f32
    out,           # (out_rows, hidden) f32 (zero-initialized)
    OUT_ROWS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    source = ct.bid(0)
    col_block = ct.bid(1)

    # Load one row index (int64)
    row = ct.load(index_arr, index=(source,), shape=(1,))

    # Load the source row's tile (f32).
    vals = ct.load(values, index=(source, col_block), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)

    # Build column offsets tile (int64)
    cols_i64 = ct.astype(ct.arange(BLOCK_H, dtype=ct.int32), ct.int64) + \
        ct.astype(col_block, ct.int64) * BLOCK_H

    # Skip -1 sentinel and OOB by clamping the row index. atomic_add's
    # check_bounds=True will drop atomic when row is outside [0, OUT_ROWS).
    # Also skip cols >= HIDDEN.
    row_idx = ct.reshape(row, (1, 1))
    col_idx = ct.reshape(cols_i64, (1, BLOCK_H))

    # For col out-of-bounds we set the column index to a value >= HIDDEN
    # (already the case) so check_bounds drops the atomic.
    ct.atomic_add(out, (row_idx, col_idx), vals)


@oracle_impl(hardware="B200", point="1bba723b", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
@oracle_impl(hardware="B200", point="154a9c83", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=8)
@oracle_impl(hardware="B200", point="b92ebe46", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
@oracle_impl(hardware="B200", point="e0a3bc30", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
@oracle_impl(hardware="B200", point="79ee2993", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
def oracle_forward(inputs, *, ZERO_BLOCK, BLOCK_H, scatter_warps):
    del ZERO_BLOCK, scatter_warps  # unused in cuTile port
    indices, values, out_shape = inputs
    out_rows = int(out_shape[0])
    hidden = int(out_shape[1])
    total_sources = int(indices.numel())

    # Views: index is (1, N) or (N,), flatten to 1D. Values is (1, N, H) etc,
    # flatten to (N, H).
    idx_flat = indices.reshape(total_sources).contiguous()
    vals_flat = values.reshape(total_sources, hidden).contiguous()

    out = torch.zeros(
        (out_rows, hidden),
        device=values.device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    grid = (total_sources, (hidden + BLOCK_H - 1) // BLOCK_H, 1)
    ct.launch(
        stream,
        grid,
        _row_scatter_add_kernel,
        (idx_flat, vals_flat, out, out_rows, hidden, BLOCK_H),
    )
    return out
