"""cuTile port of pointwise_1c8dca5e12e1: bf16+f32 add with f32 and bf16 outputs.

Storage-linear elementwise. All four tensors are channels-last with the same
strides so a 1D linear kernel over storage indices is safe.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 80
H = 56
W = 56
NUMEL = N * C * H * W


@ct.kernel
def _add_cast_kernel(
    x_ptr,           # bf16 flat
    y_ptr,           # f32 flat
    out_f32_ptr,     # f32 flat
    out_bf16_ptr,    # bf16 flat
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    y = ct.load(y_ptr, index=(pid,), shape=(BLOCK,))
    val = ct.astype(x, ct.float32) + y
    ct.store(out_f32_ptr, index=(pid,), tile=val)
    ct.store(out_bf16_ptr, index=(pid,), tile=ct.astype(val, ct.bfloat16))


@oracle_impl(hardware="B200", point="2a76d54a", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    x, y = inputs
    out_f32 = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, 1, W * C, C),
        device=x.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, 1, W * C, C),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # All four tensors share the same channels-last stride pattern, so
    # their underlying storage is packed with identical order. View each
    # as a linear buffer with stride 1 for a 1D cuTile kernel.
    x_flat = torch.as_strided(x, (NUMEL,), (1,))
    y_flat = torch.as_strided(y, (NUMEL,), (1,))
    of_flat = torch.as_strided(out_f32, (NUMEL,), (1,))
    ob_flat = torch.as_strided(out_bf16, (NUMEL,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(stream, ((NUMEL + BLOCK - 1) // BLOCK, 1, 1), _add_cast_kernel,
              (x_flat, y_flat, of_flat, ob_flat, BLOCK))
    return out_f32, out_bf16
