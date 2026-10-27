"""cuTile port of pointwise_8beb9f5d4ff7 (ALGEBRAIC_ELIMINATION): bf16 to f32 storage-linear cast.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_to_f32_kernel(input_ptr, output_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(output_ptr, index=(pid,), tile=ct.astype(x, ct.float32))


@oracle_impl(hardware="B200", point="a9384bfb", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK=1024):
    arg0_1, shape_param = inputs
    out_shape = (int(shape_param[0]), int(shape_param[1]), int(shape_param[2]))
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    # Flatten to 1D for the launch.
    numel = arg0_1.numel()
    input_flat = arg0_1.view(numel)
    output_flat = out.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _bf16_to_f32_kernel,
        (input_flat, output_flat, BLOCK),
    )
    return out
