"""cuTile port of pointwise_100a39b686e3 (in-place scalar i64 +1)."""

import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scalar_i64_add_copy_kernel(arg0):
    v = ct.load(arg0, index=(0,), shape=(1,))
    ct.store(arg0, index=(0,), tile=v + ct.full(shape=(1,), fill_value=1, dtype=ct.int64))


@oracle_impl(hardware="B200", point="896c6bb5")
def oracle_forward(inputs):
    arg0 = inputs[0]
    import torch
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _scalar_i64_add_copy_kernel, (arg0.view(1),))
    return arg0
