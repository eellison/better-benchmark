"""cuTile port of pointwise_c9af56d69f6c: Gemma 13x causal mask materialization."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 1000
HEADS = 8
BASE_SHAPE = (1, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUT_SHAPE = (1, HEADS, SEQ, SEQ)


@ct.kernel
def _causal_mask_13_kernel(
    out0, out1, out2, out3, out4, out5, out6,
    out7, out8, out9, out10, out11, out12,
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    r_tile = ct.bid(0)
    c_tile = ct.bid(1)
    rows = r_tile * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    cols = c_tile * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    rows2d = ct.reshape(rows, (BLOCK_R, 1))
    cols2d = ct.reshape(cols, (1, BLOCK_C))
    values = ct.where(cols2d <= rows2d, 0.0, -float("inf"))
    values_bf16 = ct.astype(values, ct.bfloat16)
    tile4d = ct.reshape(values_bf16, (1, 1, BLOCK_R, BLOCK_C))
    ct.store(out0, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out1, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out2, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out3, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out4, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out5, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out6, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out7, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out8, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out9, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out10, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out11, index=(0, 0, r_tile, c_tile), tile=tile4d)
    ct.store(out12, index=(0, 0, r_tile, c_tile), tile=tile4d)


@oracle_impl(hardware="B200", point="d7517139", BLOCK_R=8, BLOCK_C=8)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    del inputs
    device = torch.device("cuda")
    bases = [
        torch.empty_strided(
            BASE_SHAPE,
            BASE_STRIDE,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(13)
    ]
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(SEQ, BLOCK_R), ct.cdiv(SEQ, BLOCK_C), 1),
        _causal_mask_13_kernel,
        (*bases, BLOCK_R, BLOCK_C),
    )
    return tuple(base.expand(OUT_SHAPE) for base in bases)
