"""cuTile port of pointwise_b09beca3a7bd: 12-input bf16-to-fp32 residual accumulation."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _sum12_bf16_to_f32_kernel(
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
    out,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a0 = ct.astype(ct.load(x0, index=(pid,), shape=(BLOCK,)), ct.float32)
    a1 = ct.astype(ct.load(x1, index=(pid,), shape=(BLOCK,)), ct.float32)
    a2 = ct.astype(ct.load(x2, index=(pid,), shape=(BLOCK,)), ct.float32)
    a3 = ct.astype(ct.load(x3, index=(pid,), shape=(BLOCK,)), ct.float32)
    a4 = ct.astype(ct.load(x4, index=(pid,), shape=(BLOCK,)), ct.float32)
    a5 = ct.astype(ct.load(x5, index=(pid,), shape=(BLOCK,)), ct.float32)
    a6 = ct.astype(ct.load(x6, index=(pid,), shape=(BLOCK,)), ct.float32)
    a7 = ct.astype(ct.load(x7, index=(pid,), shape=(BLOCK,)), ct.float32)
    a8 = ct.astype(ct.load(x8, index=(pid,), shape=(BLOCK,)), ct.float32)
    a9 = ct.astype(ct.load(x9, index=(pid,), shape=(BLOCK,)), ct.float32)
    a10 = ct.astype(ct.load(x10, index=(pid,), shape=(BLOCK,)), ct.float32)
    a11 = ct.astype(ct.load(x11, index=(pid,), shape=(BLOCK,)), ct.float32)
    acc = a0 + a1
    acc = acc + a2
    acc = acc + a3
    acc = acc + a4
    acc = acc + a5
    acc = acc + a6
    acc = acc + a7
    acc = acc + a8
    acc = acc + a9
    acc = acc + a10
    acc = acc + a11
    ct.store(out, index=(pid,), tile=acc)


@oracle_impl(hardware="B200", point="26d25975", BLOCK=1024)
@oracle_impl(hardware="B200", point="69f8f171", BLOCK=1024)
@oracle_impl(hardware="B200", point="cf776752", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    tensors = inputs
    rows, cols = tensors[0].shape
    out = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=tensors[0].device,
        dtype=torch.float32,
    )
    # Flatten inputs and out to 1D for the pointwise kernel.
    flats = tuple(t.view(-1) for t in tensors[:12])
    out_flat = out.view(-1)
    n = out_flat.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _sum12_bf16_to_f32_kernel,
        (*flats, out_flat, BLOCK),
    )
    return out
