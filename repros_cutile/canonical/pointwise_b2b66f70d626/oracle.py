"""cuTile port of pointwise_b2b66f70d626: MobileBERT row/hidden affine broadcast.

Rowwise affine: x[i,j] * scale[j] + bias[j] with bf16 rounding of the mul
before the add. Returns two aliased views of the same output buffer.
Inline PTX add.rn/mul.rn in the Triton kernel are just default fp32 arithmetic
in cuTile (round-to-nearest-even).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
COLS = 128


@ct.kernel
def _rowwise_affine_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_M: ct.Constant[int],
    N_COLS: ct.Constant[int],
):
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, N_COLS))
    scale = ct.load(scale_ptr, index=(0,), shape=(N_COLS,))
    bias = ct.load(bias_ptr, index=(0,), shape=(N_COLS,))

    x_f = ct.astype(x, ct.float32)
    scale_f = ct.astype(scale, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    scale_2d = ct.reshape(scale_f, (1, N_COLS))
    bias_2d = ct.reshape(bias_f, (1, N_COLS))

    product = x_f * scale_2d
    product_bf = ct.astype(product, ct.bfloat16)
    product_bf_f = ct.astype(product_bf, ct.float32)
    y = product_bf_f + bias_2d
    y_bf = ct.astype(y, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=y_bf)


@oracle_impl(hardware="B200", point="bc0fb1fb", BLOCK_M=16)
def oracle_forward(inputs, *, BLOCK_M):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    base = torch.empty_like(arg0_1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS // BLOCK_M, 1, 1),
        _rowwise_affine_kernel,
        (arg0_1, arg1_1, arg2_1, base, BLOCK_M, COLS),
    )
    return (
        torch.ops.aten.view.default(base, _shape_param_1),
        torch.ops.aten.view.default(base, _shape_param_2),
    )
