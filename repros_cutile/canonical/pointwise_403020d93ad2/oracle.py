"""cuTile port of pointwise_403020d93ad2 (BANDWIDTH_BOUND): bool `arg < 0`
predicate over [8, 1024] input (bf16 or f32).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8
COLS = 1024
NUMEL = ROWS * COLS


@ct.kernel
def _lt_zero_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=x < 0.0)


@oracle_impl(hardware="B200", point="a7e138d5", BLOCK=1024)
@oracle_impl(hardware="B200", point="5002b631", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (arg0_1,) = inputs
    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    x_flat = arg0_1.view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK), 1, 1),
        _lt_zero_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return out
