"""cuTile port of pointwise_514cc391d8e4: bf16 + bf16 -> fp32 element-wise add."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_add_to_fp32_kernel(
    left_ptr,
    right_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    left = ct.load(left_ptr, index=(pid,), shape=(BLOCK,))
    right = ct.load(right_ptr, index=(pid,), shape=(BLOCK,))
    out = ct.astype(left, ct.float32) + ct.astype(right, ct.float32)
    ct.store(out_ptr, index=(pid,), tile=out)


def _contiguous_3d_stride(shape):
    return (int(shape[1]) * int(shape[2]), int(shape[2]), 1)


@oracle_impl(hardware="B200", point="a3b18231", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="e47867ab", BLOCK_SIZE=2048)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    left, right, out_shape, _out_shape_1 = inputs
    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=left.device,
        dtype=torch.float32,
    )
    n_elements = left.numel()
    left_flat = left.reshape(n_elements)
    right_flat = right.reshape(n_elements)
    out_flat = out.view(n_elements)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1),
        _bf16_add_to_fp32_kernel,
        (left_flat, right_flat, out_flat, BLOCK_SIZE),
    )
    return out
