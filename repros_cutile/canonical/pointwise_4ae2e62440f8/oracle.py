"""cuTile port of pointwise_4ae2e62440f8: PLBart 6x causal mask materialization."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 1024
N_OUTPUTS = 6
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@ct.kernel
def _causal_mask_6_kernel(
    out0, out1, out2, out3, out4, out5,
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    b = ct.bid(0)
    r_tile = ct.bid(1)
    c_tile = ct.bid(2)

    # arange indices within this tile
    rows = r_tile * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    cols = c_tile * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)

    rows2d = ct.reshape(rows, (BLOCK_R, 1))
    cols2d = ct.reshape(cols, (1, BLOCK_C))
    values = ct.where(cols2d <= rows2d, 0.0, -float("inf"))
    values_bf16 = ct.astype(values, ct.bfloat16)
    tile4d = ct.reshape(values_bf16, (1, 1, BLOCK_R, BLOCK_C))
    ct.store(out0, index=(b, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out1, index=(b, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out2, index=(b, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out3, index=(b, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out4, index=(b, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out5, index=(b, 0, r_tile, c_tile), tile=tile4d)


@oracle_impl(hardware="B200", point="d7517139", BLOCK_R=32, BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    (shape_param,) = inputs
    outputs = tuple(
        torch.empty_strided(
            (int(shape_param[0]), 1, int(shape_param[2]), int(shape_param[3])),
            OUT_STRIDE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(N_OUTPUTS)
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, ct.cdiv(SEQ, BLOCK_R), ct.cdiv(SEQ, BLOCK_C)),
        _causal_mask_6_kernel,
        (*outputs, BLOCK_R, BLOCK_C),
    )
    return outputs
