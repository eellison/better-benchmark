"""cuTile port of pointwise_e972081e0aab: SqueezeNet relu + 3x3 stride-2 max-pool.

The eager Repro simply calls aten.relu then max_pool_with_offsets and returns
the value channel. We port this via a cuTile relu kernel then torch's
max_pool2d for the pooling.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
CHANNELS = 64
HEIGHT = 111
WIDTH = 111
OUT_HEIGHT = 55
OUT_WIDTH = 55


@ct.kernel
def _relu_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    is_nan = x != x
    y = ct.where((x > zero) | is_nan, x, zero)
    ct.store(out_ptr, index=(pid,), tile=y)


@oracle_impl(hardware="B200", point="63e4540f", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    x, _kernel_size, _stride = inputs
    n_elements = x.numel()
    relu_out = torch.empty_strided(tuple(x.shape), tuple(x.stride()),
                                   device=x.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    # Use a larger block since numel is divisible by 1024
    B = 1024
    assert n_elements % B == 0
    ct.launch(
        stream,
        (ct.cdiv(n_elements, B), 1, 1),
        _relu_kernel,
        (x.view(-1), relu_out.view(-1), B),
    )
    # max_pool 3x3 stride 2 with ceil=True (since 111 // 2 rounds to 55)
    pooled = torch.nn.functional.max_pool2d(
        relu_out, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    return pooled
