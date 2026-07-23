"""cuTile port of pointwise_c4106fae2a64: bf16 relu + flatten.

Ports the Triton `_relu_flatten_kernel` to cuTile. NaN-preserving ReLU with
a 2D contiguous [512, 1280] output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 512
COLS = 1280
NUMEL = ROWS * COLS


@ct.kernel
def _relu_flatten_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f32 = ct.astype(x, ct.float32)
    # NaN-preserving ReLU: if x is NaN, keep NaN; else max(x, 0).
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    relu = ct.where(x_f32 != x_f32, x_f32, ct.where(x_f32 > 0.0, x_f32, zero))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(relu, ct.bfloat16))


@oracle_impl(hardware="B200", point="21720c2b", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    x, _shape = inputs
    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Flatten input for a 1D linear kernel.
    x_flat = x.view(NUMEL)
    out_flat = out.view(NUMEL)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (NUMEL // BLOCK, 1, 1), _relu_flatten_kernel, (x_flat, out_flat, BLOCK))
    return out
