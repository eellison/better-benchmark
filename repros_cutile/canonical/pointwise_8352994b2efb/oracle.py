"""cuTile port of pointwise_8352994b2efb: Longformer bias+as_strided overlapping stencil.

Ports the Triton `_longformer_transposed_stencil_kernel` to cuTile. For each
output tile (chunk, dim_block, token_block):
  - Decode head_batch, window, batch, head from chunk id.
  - Load a (BLOCK_T, 1, 1, BLOCK_D) tile from `x` viewed as (1024, 8, 12, 64).
  - Add bias, cast bf16, transpose to (BLOCK_D, BLOCK_T), store into out.
The as_strided/permute chain produces overlapping window slices at index
`source_token = window*256 + token`; the shape point is power-of-2 throughout
so the tile geometry is regular.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _longformer_stencil_kernel(
    x_ptr,        # bf16 [1024, 8, 12, 64] view of arg0_1
    bias_ptr,     # bf16 [768]
    out_ptr,      # bf16 [288, 64, 512]
    BLOCK_T: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    chunk = ct.bid(0)
    dim_block = ct.bid(1)
    token_block = ct.bid(2)

    head_batch = chunk // 3
    window = chunk - head_batch * 3
    batch = head_batch // 12
    head = head_batch - batch * 12

    # source_token = window*256 + token_block*BLOCK_T + range(BLOCK_T)
    # tile-space row index = (window*256 + token_block*BLOCK_T) / BLOCK_T
    row_tile = window * (256 // BLOCK_T) + token_block

    values = ct.load(
        x_ptr,
        index=(row_tile, batch, head, dim_block),
        shape=(BLOCK_T, 1, 1, BLOCK_D),
    )

    # bias is (768,) = (12*64). Tile size BLOCK_D. Bias tile-space index =
    # (head*64 + dim_block*BLOCK_D) / BLOCK_D = head*(64//BLOCK_D) + dim_block.
    bias_tile_index = head * (64 // BLOCK_D) + dim_block
    bias_tile = ct.load(bias_ptr, index=(bias_tile_index,), shape=(BLOCK_D,))

    values_2d = ct.reshape(values, (BLOCK_T, BLOCK_D))
    bias_2d = ct.reshape(bias_tile, (1, BLOCK_D))

    values_f = ct.astype(values_2d, ct.float32)
    bias_f = ct.astype(bias_2d, ct.float32)
    added = ct.astype(values_f + bias_f, ct.bfloat16)

    # Transpose (BLOCK_T, BLOCK_D) -> (BLOCK_D, BLOCK_T)
    transposed = ct.transpose(added)
    out_tile = ct.reshape(transposed, (1, BLOCK_D, BLOCK_T))
    ct.store(out_ptr, index=(chunk, dim_block, token_block), tile=out_tile)


@oracle_impl(hardware="B200", point="5fa3702b", BLOCK_T=64, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_T, BLOCK_D):
    x, bias, *_shape_params = inputs
    # x is (8192, 768) = (1024*8, 12*64). View as (1024, 8, 12, 64).
    x_view = x.view(1024, 8, 12, 64)

    out = torch.empty_strided(
        (288, 64, 512),
        (64 * 512, 512, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (288, 64 // BLOCK_D, 512 // BLOCK_T),
        _longformer_stencil_kernel,
        (x_view, bias, out, BLOCK_T, BLOCK_D),
    )
    return out
