"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete reshape -> bf16 ReLU -> reshape plus ReLU<=0 mask fanout in one contiguous Triton load/max/two-store pass, whereas Inductor currently lowers the metadata views and pointwise fanout through its generic pointwise scheduler for each captured shape; Inductor cannot do this today because the scheduler/codegen path has no dedicated view-only fanout specialization that treats the reshapes as pure metadata while emitting the required bf16 output and bool mask stores from one storage-linear traversal; the fix is SCHEDULER_FUSION: add a pointwise fanout lowering for view-only reshape producers that keeps one storage-linear kernel and preserves the returned output shapes and strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_mask_kernel(
    input_ptr,
    relu_out_ptr,
    mask_out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    relu = tl.where(x != x, x, tl.maximum(x, 0.0))
    tl.store(relu_out_ptr + offsets, relu, mask=mask)
    tl.store(mask_out_ptr + offsets, x <= 0.0, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="e44d982c", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="47a892ec", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, mask_shape, output_shape = inputs
    output_shape = tuple(output_shape)
    mask_shape = tuple(mask_shape)
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
    grid = (triton.cdiv(n_elements, BLOCK),)
    _relu_mask_kernel[grid](
        arg0_1,
        relu_out,
        mask_out,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return relu_out, mask_out
