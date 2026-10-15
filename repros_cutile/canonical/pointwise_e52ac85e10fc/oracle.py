"""cuTile port of pointwise_e52ac85e10fc: allocate a zero-dim int64 tensor with value 4096."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scalar_i64_full_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.full(shape=(1,), fill_value=4096, dtype=ct.int64))


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    out = torch.empty((), device=torch.device("cuda", 0), dtype=torch.int64)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _scalar_i64_full_kernel, (out.view(1),))
    return out
