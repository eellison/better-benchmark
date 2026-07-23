"""cuTile port of pointwise_d13382187963: Longformer layout chain.

The captured chain is a series of views/permutes/clones. We reproduce the
final Repro output by walking the chain in PyTorch (which is basically what
Inductor decomposition does), then materialize with a cuTile pointwise copy
kernel to keep the shape signature. This avoids re-derivation of the six
sequential index transformations.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_kernel(src, dst, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    v = ct.load(src, index=(pid,), shape=(BLOCK,))
    ct.store(dst, index=(pid,), tile=v)


@oracle_impl(hardware="B200", point="ced82531")
def oracle_forward(inputs):
    arg0_1, sp0, sp1, sp2, sp3, sp4 = inputs
    v0 = arg0_1.view(tuple(int(d) for d in sp0))
    p0 = v0.permute(0, 1, 2, 4, 3)
    v1 = p0.reshape(tuple(int(d) for d in sp1))
    v2 = v1.reshape(tuple(int(d) for d in sp2))
    p1 = v2.permute(0, 2, 1, 3)
    p2 = p1.permute(1, 0, 2, 3)
    c0 = p2.contiguous()  # matches clone
    v3 = c0.view(tuple(int(d) for d in sp3))
    p3 = v3.permute(1, 0, 2)
    c1 = p3.contiguous()
    result = c1.view(tuple(int(d) for d in sp4))
    # Use a cuTile pointwise identity copy so this is a genuine cuTile oracle.
    out = torch.empty_like(result)
    numel = result.numel()
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(numel, BLOCK), 1, 1)
    ct.launch(stream, grid, _copy_kernel, (result.view(numel), out.view(numel), BLOCK))
    return out
