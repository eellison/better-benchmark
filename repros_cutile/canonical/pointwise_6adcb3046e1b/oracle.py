"""cuTile port of pointwise_6adcb3046e1b: bf16 hard-swish flatten."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_ELEMENTS = 512 * 1280


@ct.kernel
def _hardswish_flatten_bf16_kernel(input_ptr, output_ptr, BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    x_bf = ct.load(input_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x = ct.astype(x_bf, ct.float32)
    shifted = x + 3.0
    clamp_min_v = ct.where(shifted < 0.0, ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32), shifted)
    clamp_max_v = ct.where(clamp_min_v > 6.0, ct.full((BLOCK_SIZE,), 6.0, dtype=ct.float32), clamp_min_v)
    out = (x * clamp_max_v) * (1.0 / 6.0)
    ct.store(output_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="21720c2b", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    arg0_1, _shape_param_0 = inputs
    output = torch.empty_strided(
        tuple(_shape_param_0),
        (1280, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N_ELEMENTS // BLOCK_SIZE, 1, 1),
        _hardswish_flatten_bf16_kernel,
        (arg0_1.view(N_ELEMENTS), output.view(N_ELEMENTS), BLOCK_SIZE),
    )
    return output
