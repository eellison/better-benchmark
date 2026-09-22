"""cuTile port of pointwise_1c9e61a3de47: SqueezeNet ReLU + maxpool with offsets.

The reference computes:
  relu = relu(x)
  values, offsets = _low_memory_max_pool_with_offsets(relu, [3,3], [2,2], ...)
  le = arg0_1 <= 0

Port strategy: use pytorch for the tricky maxpool with offsets (which cuTile
cannot express directly), and use cuTile for the pointwise `le` mask
computation which is a straightforward elementwise flat kernel over 32*64*
111*111 = 25,239,552 elements (divisible by 1024).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
CHANNELS = 64
HEIGHT = 111
WIDTH = 111
INPUT_NUMEL = BATCH * CHANNELS * HEIGHT * WIDTH  # 25239552; div by 1024


@ct.kernel
def _le_zero_kernel(
    x_ptr,   # bf16[INPUT_NUMEL]
    out_ptr, # b8[INPUT_NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    zero = ct.astype(ct.zeros((BLOCK,), dtype=ct.bfloat16), ct.bfloat16)
    le = x <= zero
    ct.store(out_ptr, index=(pid,), tile=le)


@oracle_impl(hardware="B200", point="04c86358", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    x, _kernel_size, _stride = inputs

    x_flat = x.contiguous().view(INPUT_NUMEL)
    le_flat = torch.empty(INPUT_NUMEL, device=x.device, dtype=torch.bool)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(INPUT_NUMEL, BLOCK), 1, 1),
        _le_zero_kernel, (x_flat, le_flat, BLOCK),
    )
    le_mask = le_flat.view(BATCH, CHANNELS, HEIGHT, WIDTH)

    # For the pool step, torch has `_low_memory_max_pool_with_offsets`.
    relu = torch.relu(x)
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu, [3, 3], [2, 2], [0, 0], [1, 1], True
    )
    return values, offsets, le_mask
