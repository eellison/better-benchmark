"""cuTile port of pointwise_0b055fee3826: allocate and zero two int64[1] tensors."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _zero_two_i64_kernel(iota_ptr, add_ptr):
    zero = ct.zeros(shape=(1,), dtype=ct.int64)
    ct.store(iota_ptr, index=(0,), tile=zero)
    ct.store(add_ptr, index=(0,), tile=zero)


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs
    device = torch.device("cuda", 0)
    iota = torch.empty_strided((1,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((1,), (1,), device=device, dtype=torch.int64)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _zero_two_i64_kernel, (iota, add))
    return iota, add
