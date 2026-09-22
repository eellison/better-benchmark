"""cuTile port of sum_2ba0992ca00b: tanh-GELU backward + column sum.

Two-pass approach:
1. Row-tile kernel computes GELU-backward tanh formula and stores per-tile
   partial column sums along with the full [rows, cols] output.
2. Reduction kernel finalizes column sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _gelu_bwd_kernel(
    grad_ptr,   # f32 [rows, cols]
    x_ptr,      # f32 [rows, cols]
    out_ptr,    # f32 [rows, cols]
    partial_ptr,  # f32 [num_row_blocks, cols]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    grad = ct.load(grad_ptr, index=(row_block, col_block), shape=(BLOCK_R, BLOCK_C))
    x = ct.load(x_ptr, index=(row_block, col_block), shape=(BLOCK_R, BLOCK_C))
    mul = x * 0.5
    mul_1 = grad * mul
    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    # tanh via exp
    e2x = ct.exp(tanh_arg * 2.0)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    add_1 = tanh_val + 1.0
    mul_4 = grad * add_1
    sub = 1.0 - tanh_val * tanh_val
    mul_7 = mul_1 * sub * 0.7978845608028654
    mul_10 = mul_7 * 0.044715 * (x2 * 3.0)
    add_2 = mul_7 + mul_10
    mul_11 = mul_4 * 0.5
    result = add_2 + mul_11

    ct.store(out_ptr, index=(row_block, col_block), tile=result)

    # Partial column sum
    partial = ct.sum(result, axis=0)  # (BLOCK_C,)
    partial_2d = ct.reshape(partial, (1, BLOCK_C))
    ct.store(partial_ptr, index=(row_block, col_block), tile=partial_2d)


@ct.kernel
def _reduce_partials_kernel(
    partial_ptr,   # f32 [num_row_blocks, cols]
    out_ptr,       # f32 [cols]
    NUM_BLOCKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_RB: ct.Constant[int],
):
    col_block = ct.bid(0)
    tile = ct.load(partial_ptr, index=(0, col_block), shape=(BLOCK_RB, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(tile, axis=0)  # (BLOCK_C,)
    ct.store(out_ptr, index=(col_block,), tile=total)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="94cc1ac5")
def oracle_forward(inputs):
    grad, x, _s0, _s1, out_shape_arg, sum_shape_arg = inputs
    rows = int(grad.shape[0])
    cols = int(grad.shape[1])

    out = torch.empty_strided(
        _shape_tuple(out_shape_arg),
        (cols, 1),
        device=grad.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        _shape_tuple(sum_shape_arg),
        (1,),
        device=grad.device,
        dtype=torch.float32,
    )

    BLOCK_R = 128
    BLOCK_C = 128
    num_row_blocks = (rows + BLOCK_R - 1) // BLOCK_R
    partials = torch.empty(
        (num_row_blocks, cols), device=grad.device, dtype=torch.float32
    )

    stream = torch.cuda.current_stream()
    out_2d = out.view(rows, cols)
    ct.launch(
        stream,
        (num_row_blocks, cols // BLOCK_C, 1),
        _gelu_bwd_kernel,
        (grad, x, out_2d, partials, BLOCK_R, BLOCK_C),
    )
    # Reduce partials.
    block_rb = 1
    while block_rb < num_row_blocks:
        block_rb *= 2
    ct.launch(
        stream,
        (cols // BLOCK_C, 1, 1),
        _reduce_partials_kernel,
        (partials, sum_out.view(cols), num_row_blocks, BLOCK_C, block_rb),
    )

    return out, out.permute(1, 0), sum_out
