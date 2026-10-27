"""cuTile port of pointwise_6a41a6a28b15: fill [8, 1024] f32 with -0.0."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUT_SHAPE = (8, 1024)
OUT_STRIDE = (1024, 1)
OUT_NUMEL = 8 * 1024


@ct.kernel
def _fill_negative_zero_kernel(out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    # -0.0 in f32 = bit pattern 0x80000000
    values = ct.bitcast(
        ct.full(shape=(BLOCK,), fill_value=-2147483648, dtype=ct.int32),
        ct.float32,
    )
    ct.store(out_ptr, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    _shape_param_0, = inputs
    del _shape_param_0
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(OUT_NUMEL, BLOCK), 1, 1),
        _fill_negative_zero_kernel,
        (out_flat, BLOCK),
    )
    return out
