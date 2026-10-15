"""cuTile port of pointwise_6ee05b1c8806: bf16 -> f32 storage-linear cast."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_to_f32_kernel(in_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    values = ct.load(in_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(values, ct.float32))


@oracle_impl(hardware="B200", point="a9384bfb", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, shape_param = inputs
    dim0 = int(shape_param[0])
    dim1 = int(shape_param[1])
    dim2 = int(shape_param[2])
    numel = arg0_1.numel()
    # Allocate a contiguous flat buffer and reinterpret with the requested strides.
    flat = torch.empty(numel, device=arg0_1.device, dtype=torch.float32)
    out = torch.as_strided(flat, (dim2, dim1, dim0), (1, dim2, dim1 * dim2))
    in_flat = arg0_1.reshape(numel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _bf16_to_f32_kernel,
        (in_flat, flat, BLOCK),
    )
    return out
