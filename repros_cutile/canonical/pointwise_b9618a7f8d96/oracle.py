"""cuTile port of pointwise_b9618a7f8d96: scalar f32 copy/mul/sqrt.

Loads the singleton input, writes copy=value, mul=value*1, sqrt=sqrt(mul).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scalar_copy_mul_sqrt_kernel(input_ptr, copy_ptr, mul_ptr, sqrt_ptr):
    value = ct.load(input_ptr, index=(0,), shape=(1,))
    mul = value * ct.full(shape=(1,), fill_value=1.0, dtype=ct.float32)
    ct.store(copy_ptr, index=(0,), tile=value)
    ct.store(mul_ptr, index=(0,), tile=mul)
    ct.store(sqrt_ptr, index=(0,), tile=ct.sqrt(mul))


@oracle_impl(hardware="B200", point="406cae07")
def oracle_forward(inputs):
    (arg0_1,) = inputs
    copy = torch.empty_like(arg0_1)
    mul = torch.empty_like(arg0_1)
    sqrt = torch.empty_like(arg0_1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _scalar_copy_mul_sqrt_kernel,
        (arg0_1.view(1), copy.view(1), mul.view(1), sqrt.view(1)),
    )
    return copy, mul, sqrt
