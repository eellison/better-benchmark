"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full bf16 `view -> relu -> view` scope as one storage-linear Triton load/select/store kernel registered for every captured dense shape point, whereas Inductor lowers the metadata-only views plus ReLU through its generic pointwise scheduler; Inductor cannot do this today because it has no B200-tuned reshape-eliding bf16 ReLU materialization template for these transformer activation layouts and relies on generic pointwise launch and indexing choices; the fix is NEW_PATTERN: add a guarded bf16 ReLU materialization template or equivalent pointwise autotuning specialization for dense contiguous activations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_view_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x < 0.0, 0.0, x)
    tl.store(output_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="e44d982c", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="b4a7958d", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="47a892ec", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    x = inputs[0]
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _relu_view_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out
