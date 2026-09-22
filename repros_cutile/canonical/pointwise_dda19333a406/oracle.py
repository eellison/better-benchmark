"""cuTile port of pointwise_dda19333a406: f32-to-bf16 cast plus permute(1, 0) view."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _f32_to_bf16_kernel(input_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(x, ct.bfloat16))


@oracle_impl(hardware="B200", point="af408fe3", BLOCK=1024)
@oracle_impl(hardware="B200", point="fddbd2f3", BLOCK=16)
def oracle_forward(inputs, *, BLOCK: int):
    (x,) = inputs
    base = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.reshape(-1)
    base_flat = base.reshape(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _f32_to_bf16_kernel,
        (x_flat, base_flat, BLOCK),
    )
    return base.permute(1, 0)
