"""cuTile port of pointwise_5e98a1498875: bf16 MobileNetV3 hard-swish pointwise."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_ELEMENTS = 32 * 1280


@ct.kernel
def _hardswish_bf16_kernel(input_ptr, output_ptr, BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    x_bf = ct.load(input_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x = ct.astype(x_bf, ct.float32)
    shifted = x + 3.0
    clamp_min = ct.where(shifted < 0.0, ct.full(shape=(BLOCK_SIZE,), fill_value=0.0, dtype=ct.float32), shifted)
    clamp_max = ct.where(clamp_min > 6.0, ct.full(shape=(BLOCK_SIZE,), fill_value=6.0, dtype=ct.float32), clamp_min)
    out = (x * clamp_max) / 6.0
    ct.store(output_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="040ff6c3", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    (arg0_1,) = inputs
    output = torch.empty_strided(
        (32, 1280),
        (1280, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    x_flat = arg0_1.view(N_ELEMENTS)
    out_flat = output.view(N_ELEMENTS)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N_ELEMENTS // BLOCK_SIZE, 1, 1),
        _hardswish_bf16_kernel,
        (x_flat, out_flat, BLOCK_SIZE),
    )
    return output
