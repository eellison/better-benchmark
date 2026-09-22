"""cuTile port of pointwise_edfab1af9465: bf16 aten.tanh via fp32 tanh with bf16 store."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _tanh_bf16_kernel(x_ptr, out_ptr, BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x_f32 = ct.astype(x, ct.float32)
    y = ct.tanh(x_f32)
    y_bf16 = ct.astype(y, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=y_bf16)


@oracle_impl(hardware="B200", point="3fee83c6", BLOCK_SIZE=256)
@oracle_impl(hardware="B200", point="5a025cf0", BLOCK_SIZE=64)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(x.numel(), BLOCK_SIZE), 1, 1),
        _tanh_bf16_kernel,
        (x.view(-1), out.view(-1), BLOCK_SIZE),
    )
    return out
