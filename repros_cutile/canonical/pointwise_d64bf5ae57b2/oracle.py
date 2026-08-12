"""cuTile port of pointwise_d64bf5ae57b2 (NEW_PATTERN): zero-valued int64[1] iota + unsqueeze view."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _zero_i64_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.int64))


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs
    iota = torch.empty_strided(
        (1,), (1,), device=torch.device("cuda", 0), dtype=torch.int64,
    )
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _zero_i64_kernel, (iota,))
    return iota, iota.unsqueeze(-1)
