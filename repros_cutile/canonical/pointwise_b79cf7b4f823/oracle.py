"""cuTile port of pointwise_b79cf7b4f823: MobileBERT affine-alias (add, mul, add)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
COLS = 512


@ct.kernel
def _affine_alias_bf16_kernel(
    arg0,       # bf16 [ROWS, COLS] (viewed as (256, 128, 512))
    arg1,       # bf16 [ROWS, COLS]
    scale,      # bf16 [COLS]
    bias,       # bf16 [COLS]
    out,        # bf16 [ROWS, COLS]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    r = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(arg0, index=(r, c), shape=(BLOCK_R, BLOCK_C))
    residual = ct.load(arg1, index=(r, c), shape=(BLOCK_R, BLOCK_C))
    scale_v = ct.load(scale, index=(c,), shape=(BLOCK_C,))
    bias_v = ct.load(bias, index=(c,), shape=(BLOCK_C,))
    scale_2d = ct.reshape(scale_v, (1, BLOCK_C))
    bias_2d = ct.reshape(bias_v, (1, BLOCK_C))
    y = (x + residual) * scale_2d + bias_2d
    ct.store(out, index=(r, c), tile=y)


@oracle_impl(hardware="B200", point="09b2e78e", BLOCK_R=8, BLOCK_C=512)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs[:4]
    shape_0, shape_1, shape_2, shape_3 = inputs[4:8]

    # arg1_1 shape is (256, 128, 512), same underlying as (32768, 512)
    arg1_2d = arg1_1.view(ROWS, COLS)

    out = torch.empty_strided(
        tuple(shape_0),
        (shape_0[1] * shape_0[2], shape_0[2], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    out_2d = out.view(ROWS, COLS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, BLOCK_R), ct.cdiv(COLS, BLOCK_C), 1),
        _affine_alias_bf16_kernel,
        (arg0_1, arg1_2d, arg2_1, arg3_1, out_2d, BLOCK_R, BLOCK_C),
    )
    return (out, out.view(shape_1), out.view(shape_2), out.view(shape_3))
