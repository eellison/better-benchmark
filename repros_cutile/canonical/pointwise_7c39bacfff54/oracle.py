"""cuTile port of pointwise_7c39bacfff54: bf16 SwiGLU pointwise (x/(exp(-x)+1) * gate)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _swiglu_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    BLOCK_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf16 = ct.load(x_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    gate = ct.load(gate_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x = ct.astype(x_bf16, ct.float32)
    activated = x / (ct.exp(-x) + 1.0)
    activated_bf16 = ct.astype(activated, ct.bfloat16)
    out = activated_bf16 * gate
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="22f70084", BLOCK_SIZE=512)
@oracle_impl(hardware="B200", point="a1f43414", BLOCK_SIZE=2048)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, gate, _shape0, _shape1, out_shape = inputs
    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.view(-1)
    gate_flat = gate.view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1),
        _swiglu_kernel,
        (x_flat, gate_flat, out_flat, BLOCK_SIZE),
    )
    return out
