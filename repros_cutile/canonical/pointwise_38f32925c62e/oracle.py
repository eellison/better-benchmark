"""cuTile port of pointwise_38f32925c62e: bf16 outer-product affine + tanh."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _outer_affine_tanh_kernel(
    rows_ptr,   # [ROWS, 1] bf16
    cols_ptr,   # [COLS, 1] bf16
    bias_ptr,   # [COLS] bf16
    out_ptr,    # [ROWS, COLS] bf16
    COLS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    row_tile = ct.bid(0)

    # Load a row-tile of shape (BLOCK_M, 1) and cast to f32
    row_vals = ct.load(rows_ptr, index=(row_tile, 0), shape=(BLOCK_M, 1))
    row_vals_f = ct.astype(row_vals, ct.float32)
    # Load cols_ptr [COLS, 1] and squeeze to [COLS]
    col_vals_2d = ct.load(cols_ptr, index=(0, 0), shape=(COLS, 1))
    col_vals_f = ct.astype(ct.reshape(col_vals_2d, (1, COLS)), ct.float32)
    bias_1d = ct.load(bias_ptr, index=(0,), shape=(COLS,))
    bias_2d = ct.reshape(ct.astype(bias_1d, ct.float32), (1, COLS))

    product = row_vals_f * col_vals_f
    affine = product + bias_2d
    rounded_bf16 = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf16, ct.float32)
    # tanh via libdevice equivalent
    tanh_val = (ct.exp(rounded_f) - ct.exp(-rounded_f)) / (ct.exp(rounded_f) + ct.exp(-rounded_f))
    result = ct.astype(tanh_val, ct.bfloat16)
    ct.store(out_ptr, index=(row_tile, 0), tile=result)


@oracle_impl(hardware="B200", point="c67d6e69", BLOCK_M=8)
def oracle_forward(inputs, *, BLOCK_M: int):
    arg0_1, arg1_1, arg2_1 = inputs
    rows = 128
    cols = 16
    out = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows // BLOCK_M, 1, 1),
        _outer_affine_tanh_kernel,
        (arg0_1, arg1_1, arg2_1, out, cols, BLOCK_M),
    )
    return out
