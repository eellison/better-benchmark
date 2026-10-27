"""cuTile port of sum_84a4b14c240d: OPT dropout materialization + column sum.

Two kernels:
1. `_masked_product_partial_sum_kernel`: bf16 masked-product with fp32 partial sums per row block.
2. `_final_sum_kernel`: reduce the partial sums along row-block dim, bf16-round the total, output as f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
SCALE = 1.1111111111111112


@ct.kernel
def _masked_product_partial_sum_kernel(
    x_ptr,          # f32 [ROWS, HIDDEN]
    mask_ptr,       # b8 [ROWS, HIDDEN]
    product_ptr,    # bf16 [ROWS, HIDDEN]
    partial_ptr,    # f32 [num_row_blocks, HIDDEN]
    SCALE_: ct.Constant[float],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_HIDDEN: ct.Constant[int],
):
    row_block = ct.bid(0)
    hidden_block = ct.bid(1)

    x = ct.load(x_ptr, index=(row_block, hidden_block), shape=(BLOCK_ROWS, BLOCK_HIDDEN))
    x_bf16 = ct.astype(x, ct.bfloat16)
    keep = ct.load(mask_ptr, index=(row_block, hidden_block), shape=(BLOCK_ROWS, BLOCK_HIDDEN))
    keep_f32 = ct.astype(keep, ct.float32)
    scale_bf16 = ct.astype(keep_f32 * SCALE_, ct.bfloat16)
    product = ct.astype(ct.astype(x_bf16, ct.float32) * ct.astype(scale_bf16, ct.float32), ct.bfloat16)

    ct.store(product_ptr, index=(row_block, hidden_block), tile=product)

    partial = ct.sum(ct.astype(product, ct.float32), axis=0)
    # partial is (BLOCK_HIDDEN,); store into partial buffer of shape (num_row_blocks, HIDDEN)
    partial_2d = ct.reshape(partial, (1, BLOCK_HIDDEN))
    ct.store(partial_ptr, index=(row_block, hidden_block), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,    # f32 [BLOCK_TILES, HIDDEN]
    sum_ptr,        # f32 [HIDDEN]
    BLOCK_TILES: ct.Constant[int],
    BLOCK_HIDDEN: ct.Constant[int],
):
    hidden_block = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, hidden_block), shape=(BLOCK_TILES, BLOCK_HIDDEN))
    summed = ct.sum(values, axis=0)
    rounded_bf16 = ct.astype(summed, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    ct.store(sum_ptr, index=(hidden_block,), tile=rounded_f32)


@oracle_impl(
    hardware="B200",
    point="0955d043",
    BLOCK_ROWS=256,
    BLOCK_HIDDEN=64,
    FINAL_BLOCK_HIDDEN=64,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_HIDDEN: int,
    FINAL_BLOCK_HIDDEN: int,
):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    view = arg0_1.view(ROWS, HIDDEN)
    product = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    num_row_blocks = (ROWS + BLOCK_ROWS - 1) // BLOCK_ROWS
    partial = torch.empty_strided(
        (num_row_blocks, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    reduced = torch.empty_strided(
        (HIDDEN,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    # Mask is bool - reinterpret as uint8 for cuTile load
    mask_u8 = arg1_1.view(torch.uint8)

    stream = torch.cuda.current_stream()
    hidden_blocks = HIDDEN // BLOCK_HIDDEN
    ct.launch(
        stream,
        (num_row_blocks, hidden_blocks, 1),
        _masked_product_partial_sum_kernel,
        (view, mask_u8, product, partial, SCALE, BLOCK_ROWS, BLOCK_HIDDEN),
    )
    # BLOCK_TILES must be power of 2. num_row_blocks = 8192/256 = 32 (power of 2).
    block_tiles = 1
    while block_tiles < num_row_blocks:
        block_tiles *= 2
    ct.launch(
        stream,
        (HIDDEN // FINAL_BLOCK_HIDDEN, 1, 1),
        _final_sum_kernel,
        (partial, reduced, block_tiles, FINAL_BLOCK_HIDDEN),
    )
    return view, product, product.permute(1, 0), reduced
