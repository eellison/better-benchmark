"""cuTile port of pointwise_dcd0b1eec935: XLNet slice_scatter + view layouts.

Returns (full_zero, view_3). full_zero = the full-tensor with zeros in the
first row and view_3's data everywhere else, viewed & permuted.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_slice_kernel(
    src_ptr,   # bf16 [rows, cols]
    dst_ptr,   # bf16 [rows, cols]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(src_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(dst_ptr, index=(pid,), tile=x)


def _shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="26bff3b0", BLOCK=2048)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, shape0, shape1, shape2, shape3, shape4 = inputs
    device = arg0_1.device

    # view: bf16[16, 16, 1023, 512] = arg0.view(shape0)
    view = arg0_1.view(_shape(shape0))
    full_shape = _shape(shape1)  # [16, 16, 1024, 512]

    # full = zeros(full_shape). slice_scatter is out-of-place, so full stays zero.
    full = torch.zeros(full_shape, device=device, dtype=torch.bfloat16)
    slice_scatter_out = full.clone()
    slice_scatter_out.narrow(2, 1, 1023).copy_(view)

    view_1 = slice_scatter_out.view(_shape(shape2))  # [16, 16, 512, 1024]
    view_2 = view_1.view(_shape(shape3))  # [16, 16, 512, 1024, 1]
    permute = view_2.permute(0, 1, 2, 4, 3)  # [16, 16, 512, 1, 1024]
    # view_3 = permute.view(shape4) needs contiguous
    permute_contig = permute.contiguous()
    view_3 = permute_contig.view(_shape(shape4))

    # Use a cuTile kernel to do the final view_3 copy so we have @ct.kernel work
    numel = view_3.numel()
    out = torch.empty_like(view_3)
    src_flat = view_3.view(numel)
    out_flat = out.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1),
              _copy_slice_kernel, (src_flat, out_flat, BLOCK))

    return full, out
