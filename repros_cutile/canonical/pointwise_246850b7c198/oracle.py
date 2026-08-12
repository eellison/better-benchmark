"""cuTile port of pointwise_246850b7c198: bf16 sigmoid pointwise."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _sigmoid_kernel(
    input_ptr,
    output_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    y = 1.0 / (1.0 + ct.exp(-x_f))
    ct.store(output_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="c06edc2f", BLOCK_SIZE=256)
@oracle_impl(hardware="B200", point="c60f6058", BLOCK_SIZE=256)
@oracle_impl(hardware="B200", point="dda9fef3", BLOCK_SIZE=32)
def oracle_forward(inputs, *, BLOCK_SIZE):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.reshape(n_elements)
    out_flat = out.view(n_elements)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1),
        _sigmoid_kernel,
        (x_flat, out_flat, BLOCK_SIZE),
    )
    return out
