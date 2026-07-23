"""cuTile port of pointwise_aa77af8fcd54: bf16 gated tanh-approximate GELU."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _gated_tanh_gelu_kernel(x_ptr, gate_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    gate = ct.load(gate_ptr, index=(pid,), shape=(BLOCK,))
    xf = ct.astype(x, ct.float32)
    x2 = xf * xf
    x3 = x2 * xf
    tanh_arg = (xf + x3 * 0.044715) * 0.7978845608028654
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1); use ct.exp
    # Or approximate with tanh via 1 - 2/(exp(2x)+1)
    e = ct.exp(tanh_arg * 2.0)
    tanh = (e - 1.0) / (e + 1.0)
    gelu = (xf * 0.5) * (tanh + 1.0)
    gelu_bf16 = ct.astype(gelu, ct.bfloat16)
    y = ct.astype(ct.astype(gelu_bf16, ct.float32) * ct.astype(gate, ct.float32),
                  ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=y)


@oracle_impl(hardware="B200", point="4c39a052", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="5e420cbf", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, gate, _s0, _s1, _s2 = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n = x.numel()
    x_flat = torch.as_strided(x, (n,), (1,))
    gate_flat = torch.as_strided(gate, (n,), (1,))
    out_flat = torch.as_strided(out, (n,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(n, BLOCK_SIZE), 1, 1),
              _gated_tanh_gelu_kernel,
              (x_flat, gate_flat, out_flat, BLOCK_SIZE))
    return out
