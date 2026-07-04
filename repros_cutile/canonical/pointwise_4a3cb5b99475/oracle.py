"""cuTile port of pointwise_4a3cb5b99475: bf16 ReLU with bool mask fanout.

SCHEDULER_FUSION: one storage-linear pass reads the contiguous bf16 input,
writes ReLU(x) into the relu output (with NaN-preserving semantics), and the
bool mask x<=0 into a sibling bool output. NaNs propagate through ReLU
because we test `x != x`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_mask_kernel(
    x_ptr,        # bf16 [N]
    relu_ptr,     # bf16 [N]
    mask_ptr,     # bool [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    zero = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    is_nan = x != x
    # NaN-preserving relu: if NaN return NaN, else max(x, 0)
    positive = ct.where(x > zero, x, zero)
    relu = ct.where(is_nan, x, positive)
    ct.store(relu_ptr, index=(pid,), tile=relu)
    mask = x <= zero
    ct.store(mask_ptr, index=(pid,), tile=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="e44d982c", BLOCK=1024)
@oracle_impl(hardware="B200", point="47a892ec", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, mask_shape, output_shape = inputs
    output_shape = tuple(int(d) for d in output_shape)
    mask_shape = tuple(int(d) for d in mask_shape)
    relu_out = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    mask_out = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    n_elements = arg0_1.numel()
    x_flat = arg0_1.reshape(n_elements)
    relu_flat = relu_out.view(n_elements)
    mask_flat = mask_out.view(n_elements)
    grid = ((n_elements + BLOCK - 1) // BLOCK, 1, 1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _relu_mask_kernel,
        (x_flat, relu_flat, mask_flat, BLOCK),
    )
    return relu_out, mask_out
