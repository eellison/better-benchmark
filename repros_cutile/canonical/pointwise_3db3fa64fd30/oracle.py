"""cuTile port of pointwise_3db3fa64fd30: aten.full([4, 2048], 1, f32) fill."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fill_one_f32_kernel(out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    ones = ct.full(shape=(BLOCK,), fill_value=1.0, dtype=ct.float32)
    ct.store(out_ptr, index=(pid,), tile=ones)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (shape_param,) = inputs
    shape = tuple(int(dim) for dim in shape_param)
    out = torch.empty_strided(
        shape,
        (shape[1], 1),
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    total = out.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _fill_one_f32_kernel,
        (out.view(total), BLOCK),
    )
    return out
