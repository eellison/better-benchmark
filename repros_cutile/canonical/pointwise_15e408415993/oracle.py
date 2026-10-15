"""cuTile port of pointwise_15e408415993: bf16 SiLU pointwise.

Computes y = x / (exp(-x) + 1) in fp32, then bf16 cast, over a flat 1D tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _silu_kernel(x_ptr, out_ptr, BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x_f = ct.astype(x, ct.float32)
    y = x_f / (ct.exp(-x_f) + 1.0)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="fd0f60cd", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="4bf70c65", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="7f1f1e95", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    input_tensor, _shape0, output_shape = inputs
    output_shape = tuple(int(dim) for dim in output_shape)
    out = torch.empty_strided(
        output_shape,
        (output_shape[1], 1),
        device=input_tensor.device,
        dtype=torch.bfloat16,
    )
    numel = input_tensor.numel()
    flat_in = input_tensor.view(numel)
    flat_out = out.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (numel // BLOCK_SIZE, 1, 1),
        _silu_kernel,
        (flat_in, flat_out, BLOCK_SIZE),
    )
    return out
