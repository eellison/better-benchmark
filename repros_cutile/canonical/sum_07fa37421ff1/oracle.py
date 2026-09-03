"""cuTile port of sum_07fa37421ff1: Longformer bf16 clone + column sum.

The intricate strided crop + double permute + clone is done via torch (which
matches what Inductor does anyway); we then run a cuTile column-sum kernel
over the materialized `[8192, 768]` tensor to produce the returned fp32 vector.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _col_sum_kernel(
    src_ptr,       # (ROWS, COLS) bf16
    partial_ptr,   # (num_row_tiles, COLS) f32
    ROWS: ct.Constant[int],
    COLS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    row_block = ct.bid(0)
    d_block = ct.bid(1)
    tile = ct.load(src_ptr, index=(row_block, d_block), shape=(BLOCK_M, BLOCK_D))
    tile_f = ct.astype(tile, ct.float32)
    partial = ct.sum(tile_f, axis=0)
    ct.store(partial_ptr, index=(row_block, d_block), tile=ct.reshape(partial, (1, BLOCK_D)))


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr,  # (num_row_tiles, COLS) f32
    out_ptr,      # (COLS,) f32
    ROW_TILES: ct.Constant[int],
    COLS: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    d_block = ct.bid(0)
    tile = ct.load(partial_ptr, index=(0, d_block), shape=(ROW_TILES, BLOCK_D))
    acc = ct.sum(tile, axis=0)
    # Round through bf16 then back to f32 to match the Triton epilogue.
    rounded = ct.astype(ct.astype(acc, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(d_block,), tile=rounded)


# a120887f: (T([9437184], bf16), S([96,1536,64]), S([98304,64,1]), ...)
@oracle_impl(hardware="B200", point="a120887f", BLOCK_M=32, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_D: int):
    arg0_1, *_shape_params = inputs
    device = arg0_1.device

    # Materialize the cropped/permuted [8192, 768] bf16 clone via torch.
    as_strided = torch.as_strided(arg0_1, (96, 1536, 64), (98304, 64, 1), 0)
    cropped = as_strided[:, 256 : 1536 - 256, :]  # [96, 1024, 64]
    view4 = cropped.reshape(8, 12, 1024, 64)
    perm1 = view4.permute(0, 2, 1, 3)  # [8, 1024, 12, 64]
    perm2 = perm1.permute(1, 0, 2, 3)  # [1024, 8, 12, 64]
    clone = perm2.contiguous()
    out = clone.view(1024, 8, 768)
    view3 = out.view(8192, 768)

    # Column-sum kernel.
    ROWS = 8192
    COLS = 768
    row_tiles = ROWS // BLOCK_M
    d_tiles = COLS // BLOCK_D
    partial = torch.empty((row_tiles, COLS), device=device, dtype=torch.float32)
    sum_out = torch.empty((COLS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (row_tiles, d_tiles, 1),
        _col_sum_kernel,
        (view3, partial, ROWS, COLS, BLOCK_M, BLOCK_D),
    )
    ct.launch(
        stream,
        (d_tiles, 1, 1),
        _finalize_sum_kernel,
        (partial, sum_out, row_tiles, COLS, BLOCK_D),
    )
    return out.view(8192, 768), out.view(8192, 768).permute(1, 0), sum_out
