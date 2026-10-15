"""cuTile port of pointwise_00686ee773b3: bf16 -> fp32 conversion of flat tensor."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_to_f32_kernel(
    input_ptr,
    output_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.load(input_ptr, index=(pid,), shape=(BLOCK,))
    values_f32 = ct.astype(values, ct.float32)
    ct.store(output_ptr, index=(pid,), tile=values_f32)


@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK=8192)
@oracle_impl(hardware="B200", point="90358d5b", BLOCK=8192)
@oracle_impl(hardware="B200", point="bd432928", BLOCK=8192)
@oracle_impl(hardware="B200", point="07bfd41e", BLOCK=8192)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK=8192)
@oracle_impl(hardware="B200", point="b642f4d6", BLOCK=8192)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK=8192)
def oracle_forward(inputs, *, BLOCK: int):
    x, shape_param = inputs
    dim0 = int(shape_param[0])
    dim1 = int(shape_param[1])
    dim2 = int(shape_param[2])
    output = torch.empty_strided(
        (dim0, dim1, dim2),
        (dim1 * dim2, dim2, 1),
        device=x.device,
        dtype=torch.float32,
    )
    x_flat = x.view(-1)
    out_flat = output.view(-1)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _bf16_to_f32_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return output
