"""cuTile port of sum_42c689cc9c14: Longformer bf16 scale + hidden-dim sum.

Two cuTile kernels:
  1. Scale-and-partial-sum: each (BLOCK_ROWS x BLOCK_HIDDEN) tile of the
     [8192, 768] scaled bf16 tensor is written; partial fp32 column sums are
     stored per row-block.
  2. Final-sum: for each hidden column tile, sum all row-block partials and
     round through bf16 to fp32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
SCALE = 0.125


@ct.kernel
def _scale_and_partial_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN] (view of arg0_1)
    out_ptr,        # bf16 [ROWS, HIDDEN]
    partial_ptr,    # f32 [num_row_blocks, HIDDEN]
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_HIDDEN: ct.Constant[int],
):
    row_block = ct.bid(0)
    hidden_block = ct.bid(1)

    x = ct.load(x_ptr, index=(row_block, hidden_block),
                shape=(BLOCK_ROWS, BLOCK_HIDDEN))
    x_f = ct.astype(x, ct.float32)
    scaled = ct.astype(x_f * SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, hidden_block), tile=scaled)

    scaled_f = ct.astype(scaled, ct.float32)
    partial = ct.sum(scaled_f, axis=0)  # (BLOCK_HIDDEN,)
    partial_2d = ct.reshape(partial, (1, BLOCK_HIDDEN))
    ct.store(partial_ptr, index=(row_block, hidden_block), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,    # f32 [num_row_blocks, HIDDEN]
    out_ptr,        # f32 [HIDDEN]
    BLOCK_TILES: ct.Constant[int],
    BLOCK_HIDDEN: ct.Constant[int],
):
    hidden_block = ct.bid(0)
    tile = ct.load(
        partial_ptr,
        index=(0, hidden_block),
        shape=(BLOCK_TILES, BLOCK_HIDDEN),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(tile, axis=0)  # (BLOCK_HIDDEN,)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(hidden_block,), tile=rounded)


@oracle_impl(hardware="B200", point="37e882df", BLOCK_ROWS=128, BLOCK_HIDDEN=64)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_HIDDEN):
    x, *_shape_params = inputs
    x_2d = x.view(ROWS, HIDDEN)

    scaled = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    num_row_blocks = ROWS // BLOCK_ROWS
    partial = torch.empty(
        (num_row_blocks, HIDDEN),
        device=x.device,
        dtype=torch.float32,
    )
    reduced = torch.empty((HIDDEN,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_blocks, HIDDEN // BLOCK_HIDDEN, 1),
        _scale_and_partial_kernel,
        (x_2d, scaled, partial, BLOCK_ROWS, BLOCK_HIDDEN),
    )
    # Round num_row_blocks up to next power of 2 for the final reduction tile.
    block_tiles = 1
    while block_tiles < num_row_blocks:
        block_tiles *= 2
    ct.launch(
        stream,
        (HIDDEN // BLOCK_HIDDEN, 1, 1),
        _final_sum_kernel,
        (partial, reduced, block_tiles, BLOCK_HIDDEN),
    )
    return scaled, scaled.permute(1, 0), reduced
