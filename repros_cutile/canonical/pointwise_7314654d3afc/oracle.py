"""cuTile port of pointwise_7314654d3afc: bf16 (x+y)/2 with bf16 rounding boundary."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_ELEMENTS = 128 * 1000


@ct.kernel
def _add_div_bf16_kernel(x_ptr, y_ptr, out_ptr, BLOCK_N: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,)), ct.float32)
    y = ct.astype(ct.load(y_ptr, index=(pid,), shape=(BLOCK_N,)), ct.float32)
    summed = ct.astype(x + y, ct.bfloat16)
    out = ct.astype(ct.astype(summed, ct.float32) / 2.0, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="ced5d5ea", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, y = inputs
    out = torch.empty_like(x)
    n = x.numel()
    x_flat = x.view(n)
    y_flat = y.view(n)
    out_flat = out.view(n)
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n, BLOCK_N), 1, 1)
    ct.launch(
        stream,
        grid,
        _add_div_bf16_kernel,
        (x_flat, y_flat, out_flat, BLOCK_N),
    )
    return out
