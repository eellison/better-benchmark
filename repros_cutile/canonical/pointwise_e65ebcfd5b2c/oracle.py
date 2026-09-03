"""cuTile port of pointwise_e65ebcfd5b2c: DCGAN two-input leaky-gate."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _leaky_gate_kernel(value_ptr, gate_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    v = ct.load(value_ptr, index=(pid,), shape=(BLOCK,))
    g = ct.load(gate_ptr, index=(pid,), shape=(BLOCK,))
    vf = ct.astype(v, ct.float32)
    gf = ct.astype(g, ct.float32)
    neg = vf * 0.2
    out = ct.where(gf > 0.0, vf, neg)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="78d93924", BLOCK=4096)
def oracle_forward(inputs, *, BLOCK: int):
    value, gate = inputs
    out = torch.empty_strided(
        tuple(value.shape),
        tuple(value.stride()),
        device=value.device,
        dtype=torch.bfloat16,
    )
    n = value.numel()
    v_flat = torch.as_strided(value, (n,), (1,))
    g_flat = torch.as_strided(gate, (n,), (1,))
    o_flat = torch.as_strided(out, (n,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(n, BLOCK), 1, 1),
              _leaky_gate_kernel, (v_flat, g_flat, o_flat, BLOCK))
    return out
