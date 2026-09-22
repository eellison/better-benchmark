"""cuTile port of pointwise_38c88cc47464: Longformer permute[1,0,2] + fp32->bf16 cast.

Reads (8, 1024, 768) f32 and writes it to (1024, 8, 768) bf16 contiguous storage.
Returns the flat (8192, 768) view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HIDDEN = 768
ROWS = BATCH * SEQ


@ct.kernel
def _permute_cast_kernel(
    src,       # f32 [BATCH, SEQ, HIDDEN]
    dst,       # bf16 [SEQ, BATCH, HIDDEN]
    BLOCK: ct.Constant[int],
):
    b = ct.bid(0)
    s = ct.bid(1)
    h_tile = ct.bid(2)
    # Load (1, 1, BLOCK) from src at (b, s, h_tile)
    values = ct.load(src, index=(b, s, h_tile), shape=(1, 1, BLOCK))
    bf = ct.astype(values, ct.bfloat16)
    ct.store(dst, index=(s, b, h_tile), tile=bf)


@oracle_impl(hardware="B200", point="784a7239", BLOCK=256)
def oracle_forward(inputs, *, BLOCK):
    x, _shape_param_0 = inputs
    out_base = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, SEQ, ct.cdiv(HIDDEN, BLOCK)),
        _permute_cast_kernel,
        (x, out_base, BLOCK),
    )
    return out_base.view(ROWS, HIDDEN)
