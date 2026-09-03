"""cuTile port of pointwise_fa1cc97c6cfd: prefix-slice bf16 -> fp32 cast."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _prefix_slice_bf16_to_f32_kernel(
    in_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.load(in_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(values, ct.float32))


# 0473776d: (T([50272,1024], bf16))
@oracle_impl(hardware="B200", point="0473776d", BLOCK=1024)
# c7a73141: (T([50272,768], bf16))
@oracle_impl(hardware="B200", point="c7a73141", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (arg0_1,) = inputs
    rows = int(arg0_1.shape[0]) - 7
    cols = int(arg0_1.shape[1])
    n = rows * cols
    # Slice keeps strides — first `rows` rows of the input tensor.
    src = arg0_1[:rows].contiguous().view(n)
    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_view = output.view(n)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _prefix_slice_bf16_to_f32_kernel,
        (src, out_view, BLOCK),
    )
    return output
