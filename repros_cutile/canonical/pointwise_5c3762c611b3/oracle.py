"""cuTile port of pointwise_5c3762c611b3: MobileBERT residual column-affine chain.

For each row: mul = (x1 * scale0) bf16; add = (mul + bias0) bf16;
add_1 = (x0 + add) bf16; mul_1 = (add_1 * scale1) bf16; out = mul_1 + bias1 bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
HIDDEN = 128
FLAT_SHAPE = (ROWS, HIDDEN)
FLAT_STRIDE = (HIDDEN, 1)
VIEW_SHAPE = (256, 128, 128)
VIEW_STRIDE = (128 * 128, 128, 1)


@ct.kernel
def _residual_column_affine_kernel(
    x0_ptr,      # bf16 [rows, HIDDEN]
    x1_ptr,      # bf16 [rows, HIDDEN]
    s0_ptr,      # bf16 [HIDDEN]
    b0_ptr,      # bf16 [HIDDEN]
    s1_ptr,      # bf16 [HIDDEN]
    b1_ptr,      # bf16 [HIDDEN]
    out_ptr,     # bf16 [rows, HIDDEN]
    BLOCK_M: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
):
    row_tile = ct.bid(0)

    x0 = ct.load(x0_ptr, index=(row_tile, 0), shape=(BLOCK_M, HIDDEN_C))
    x1 = ct.load(x1_ptr, index=(row_tile, 0), shape=(BLOCK_M, HIDDEN_C))
    s0 = ct.load(s0_ptr, index=(0,), shape=(HIDDEN_C,))
    b0 = ct.load(b0_ptr, index=(0,), shape=(HIDDEN_C,))
    s1 = ct.load(s1_ptr, index=(0,), shape=(HIDDEN_C,))
    b1 = ct.load(b1_ptr, index=(0,), shape=(HIDDEN_C,))

    x0_f = ct.astype(x0, ct.float32)
    x1_f = ct.astype(x1, ct.float32)
    s0_2d = ct.reshape(ct.astype(s0, ct.float32), (1, HIDDEN_C))
    b0_2d = ct.reshape(ct.astype(b0, ct.float32), (1, HIDDEN_C))
    s1_2d = ct.reshape(ct.astype(s1, ct.float32), (1, HIDDEN_C))
    b1_2d = ct.reshape(ct.astype(b1, ct.float32), (1, HIDDEN_C))

    mul = ct.astype(ct.astype(x1_f * s0_2d, ct.bfloat16), ct.float32)
    add = ct.astype(ct.astype(mul + b0_2d, ct.bfloat16), ct.float32)
    add_1 = ct.astype(ct.astype(x0_f + add, ct.bfloat16), ct.float32)
    mul_1 = ct.astype(ct.astype(add_1 * s1_2d, ct.bfloat16), ct.float32)
    add_2 = ct.astype(mul_1 + b1_2d, ct.bfloat16)

    ct.store(out_ptr, index=(row_tile, 0), tile=add_2)


@oracle_impl(hardware="B200", point="b07f4689")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, *_ = inputs

    out_flat = torch.empty_strided(
        FLAT_SHAPE,
        FLAT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    BLOCK_M = 16
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS // BLOCK_M, 1, 1),
        _residual_column_affine_kernel,
        (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, out_flat, BLOCK_M, HIDDEN),
    )
    return out_flat.as_strided(VIEW_SHAPE, VIEW_STRIDE), out_flat
