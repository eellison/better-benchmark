"""cuTile port of pointwise_76ffbcc3b8af: AlexNet ReLU + 3x3 stride-2 maxpool + flatten.

The 3x3 stride-2 maxpool window is a fixed data-independent stencil. A
cuTile kernel produces per-output element the max of the 9-element window
using `ct.gather` on a flat [BATCH*CHANNELS, HW_IN] view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 256
H_IN = 13
W_IN = 13
H_OUT = 6
W_OUT = 6
FLAT = CHANNELS * H_OUT * W_OUT
TOTAL = BATCH * FLAT


@ct.kernel
def _relu_maxpool_flat_kernel(
    input_ptr,   # bf16 [BATCH*CHANNELS, HW_IN]
    output_ptr,  # bf16 [BATCH * FLAT]
    W_IN_C: ct.Constant[int],
    W_OUT_C: ct.Constant[int],
    H_OUT_C: ct.Constant[int],
    CHANNELS_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lanes = ct.arange(BLOCK, dtype=ct.int32)
    linear = pid * BLOCK + lanes
    ow = linear % W_OUT_C
    tmp = linear // W_OUT_C
    oh = tmp % H_OUT_C
    tmp = tmp // H_OUT_C
    ch = tmp % CHANNELS_C
    n = tmp // CHANNELS_C

    ih0 = oh * 2
    iw0 = ow * 2

    row_idx = n * CHANNELS_C + ch

    def _gather(dh, dw):
        col = (ih0 + dh) * W_IN_C + (iw0 + dw)
        return ct.gather(input_ptr, (row_idx, col))

    x00 = ct.astype(_gather(0, 0), ct.float32)
    x01 = ct.astype(_gather(0, 1), ct.float32)
    x02 = ct.astype(_gather(0, 2), ct.float32)
    x10 = ct.astype(_gather(1, 0), ct.float32)
    x11 = ct.astype(_gather(1, 1), ct.float32)
    x12 = ct.astype(_gather(1, 2), ct.float32)
    x20 = ct.astype(_gather(2, 0), ct.float32)
    x21 = ct.astype(_gather(2, 1), ct.float32)
    x22 = ct.astype(_gather(2, 2), ct.float32)

    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32)
    x00 = ct.where(x00 > 0.0, x00, zero)
    x01 = ct.where(x01 > 0.0, x01, zero)
    x02 = ct.where(x02 > 0.0, x02, zero)
    x10 = ct.where(x10 > 0.0, x10, zero)
    x11 = ct.where(x11 > 0.0, x11, zero)
    x12 = ct.where(x12 > 0.0, x12, zero)
    x20 = ct.where(x20 > 0.0, x20, zero)
    x21 = ct.where(x21 > 0.0, x21, zero)
    x22 = ct.where(x22 > 0.0, x22, zero)

    def _mx(a, b):
        return ct.where(a > b, a, b)

    best = _mx(x00, x01)
    best = _mx(best, x02)
    best = _mx(best, x10)
    best = _mx(best, x11)
    best = _mx(best, x12)
    best = _mx(best, x20)
    best = _mx(best, x21)
    best = _mx(best, x22)

    ct.store(output_ptr, index=(pid,), tile=ct.astype(best, ct.bfloat16))


@oracle_impl(hardware="B200", point="a8ee30c6")
def oracle_forward(inputs):
    x, _kernel_size, _stride, flat_shape = inputs
    device = x.device
    BLOCK = 128

    x_2d = x.contiguous().view(BATCH * CHANNELS, H_IN * W_IN)

    out = torch.empty_strided(
        tuple(int(dim) for dim in flat_shape),
        (FLAT, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_1d = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _relu_maxpool_flat_kernel,
        (x_2d, out_1d, W_IN, W_OUT, H_OUT, CHANNELS, BLOCK),
    )
    return out
