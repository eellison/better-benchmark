"""cuTile port of pointwise_ae7e5786852b: f32 -> bf16 cast then squeeze/permute aliases."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _f32_to_bf16_kernel(input_ptr, output_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(output_ptr, index=(pid,), tile=ct.astype(x, ct.bfloat16))


@oracle_impl(hardware="B200", point="d102a86e", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, shape_param = inputs
    rows = int(shape_param[1])
    cols = int(shape_param[2])
    base = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    numel = arg0_1.numel()
    stream = torch.cuda.current_stream()
    # Use the flat view for both input and output.
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _f32_to_bf16_kernel,
        (arg0_1.view(numel), base.view(numel), BLOCK),
    )
    return base, base.permute(1, 0)
