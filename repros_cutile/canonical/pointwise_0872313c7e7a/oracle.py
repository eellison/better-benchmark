"""cuTile port of pointwise_0872313c7e7a: bf16 DCGAN leaky-ReLU (x, x*0.2)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _leaky_relu_bf16_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    y = ct.where(x_f > 0.0, x_f, x_f * 0.2)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="eb973076", BLOCK=1024)
@oracle_impl(hardware="B200", point="356e4166", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.view(n_elements)
    out_flat = out.view(n_elements)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _leaky_relu_bf16_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return out
