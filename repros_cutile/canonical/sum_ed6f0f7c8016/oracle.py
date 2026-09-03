"""cuTile port of sum_ed6f0f7c8016: bf16 dropout-mask product + column sum.

For each (M, N) shape point:
1. Compute product_bf16 = bf16(fp32_values) * bf16(bool_mask) * bf16(1.1111...)
   All done in bf16 with rounding at each step.
2. Column sum reduced across rows: bf16 round-trip then fp32.
3. Return (product_bf16, product_bf16.permute(1,0), fp32_sum).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _product_partial_kernel(
    values_ptr,   # f32 [M, N]
    keep_ptr,     # bool [M, N]
    product_ptr,  # bf16 [M, N]
    partial_ptr,  # f32 [NUM_ROW_BLOCKS, N]
    ROW_BLOCK: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)

    values_f32 = ct.load(values_ptr, index=(row_tile, col_tile),
                         shape=(ROW_BLOCK, BLOCK_N))
    values_bf16 = ct.astype(values_f32, ct.bfloat16)
    keep = ct.load(keep_ptr, index=(row_tile, col_tile),
                   shape=(ROW_BLOCK, BLOCK_N))
    keep_f32 = ct.astype(keep, ct.float32)
    keep_bf16 = ct.astype(keep_f32, ct.bfloat16)
    scale_bf16 = ct.astype(
        ct.full(shape=(ROW_BLOCK, BLOCK_N),
                 fill_value=DROPOUT_SCALE, dtype=ct.float32),
        ct.bfloat16,
    )
    keep_scaled = ct.astype(
        ct.astype(keep_bf16, ct.float32)
        * ct.astype(scale_bf16, ct.float32),
        ct.bfloat16,
    )
    product_bf16 = ct.astype(
        ct.astype(values_bf16, ct.float32)
        * ct.astype(keep_scaled, ct.float32),
        ct.bfloat16,
    )
    ct.store(product_ptr, index=(row_tile, col_tile), tile=product_bf16)

    partial = ct.sum(ct.astype(product_bf16, ct.float32),
                     axis=0, keepdims=True)
    ct.store(partial_ptr, index=(row_tile, col_tile), tile=partial)


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr, out_ptr,
    NUM_ROW_BLOCKS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)
    partials = ct.load(partial_ptr, index=(0, col_tile),
                       shape=(NUM_ROW_BLOCKS, BLOCK_N))
    summed = ct.sum(partials, axis=0, keepdims=False)
    summed_bf16 = ct.astype(summed, ct.bfloat16)
    summed_f32 = ct.astype(summed_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=summed_f32)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="e6479180",
             ROW_BLOCK=128, BLOCK_N=128, FINAL_BLOCK_N=128)
@oracle_impl(hardware="B200", point="dcc62d32",
             ROW_BLOCK=128, BLOCK_N=128, FINAL_BLOCK_N=128)
@oracle_impl(hardware="B200", point="a5b2be60",
             ROW_BLOCK=128, BLOCK_N=64, FINAL_BLOCK_N=64)
@oracle_impl(hardware="B200", point="bf9ec2f8",
             ROW_BLOCK=128, BLOCK_N=64, FINAL_BLOCK_N=64)
@oracle_impl(hardware="B200", point="14e0b026",
             ROW_BLOCK=128, BLOCK_N=32, FINAL_BLOCK_N=32)
@oracle_impl(hardware="B200", point="8d0ca3a0",
             ROW_BLOCK=128, BLOCK_N=128, FINAL_BLOCK_N=128)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_N: int, FINAL_BLOCK_N: int):
    values, keep, product_shape, sum_shape = inputs
    m, n = _shape_tuple(product_shape)
    (sum_n,) = _shape_tuple(sum_shape)

    product = torch.empty_strided((m, n), (n, 1),
                                  device=values.device, dtype=torch.bfloat16)
    num_row_blocks = (m + ROW_BLOCK - 1) // ROW_BLOCK
    partials = torch.empty_strided(
        (num_row_blocks, n), (n, 1),
        device=values.device, dtype=torch.float32,
    )
    summed = torch.empty_strided((sum_n,), (1,),
                                  device=values.device, dtype=torch.float32)
    values_2d = values.view(m, n)
    keep_2d = keep.view(m, n)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_blocks, n // BLOCK_N, 1),
        _product_partial_kernel,
        (values_2d, keep_2d, product, partials, ROW_BLOCK, BLOCK_N),
    )
    ct.launch(
        stream, (n // FINAL_BLOCK_N, 1, 1),
        _finalize_sum_kernel,
        (partials, summed, num_row_blocks, FINAL_BLOCK_N),
    )
    return product, product.permute(1, 0), summed
