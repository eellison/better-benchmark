"""cuTile port of pointwise_39610fd5aba3: contiguous f32 -> bf16 cast."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _f32_to_bf16_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    values = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(values, ct.bfloat16))


@oracle_impl(hardware="B200", point="d102a86e", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, shape_param = inputs
    rows = int(shape_param[1])
    cols = int(shape_param[2])
    output = torch.empty_strided(
        (rows, cols), (cols, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    # View source as flat since input is contiguous ranked tensor
    x_flat = arg0_1.contiguous().view(-1)
    out_flat = output.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(x_flat.numel(), BLOCK), 1, 1),
              _f32_to_bf16_kernel, (x_flat, out_flat, BLOCK))
    return output
