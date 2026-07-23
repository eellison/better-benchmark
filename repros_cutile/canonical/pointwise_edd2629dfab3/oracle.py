"""cuTile port of pointwise_edd2629dfab3: bf16 relu-view."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_view_kernel(input_ptr, output_ptr, BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    zero = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.bfloat16)
    y = ct.where(x < zero, zero, x)
    ct.store(output_ptr, index=(pid,), tile=y)


@oracle_impl(hardware="B200", point="e44d982c", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="b4a7958d", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="47a892ec", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x = inputs[0]
    out = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1),
        _relu_view_kernel,
        (x.view(n_elements), out.view(n_elements), BLOCK_SIZE),
    )
    return out
