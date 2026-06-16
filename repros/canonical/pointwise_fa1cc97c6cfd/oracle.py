"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete leading-row slice plus bf16-to-fp32 cast scope as one storage-linear Triton kernel that reads only the retained contiguous prefix rows and writes the fresh contiguous fp32 output, whereas Inductor lowers the captured `aten.slice.Tensor` feeding `prims.convert_element_type` through generic pointwise materialization; Inductor cannot do this today because its pointwise/layout scheduler does not have a guarded B200 template for prefix-slice widening casts that erases the slice indexing scaffold while preserving the exact fp32 cast boundary; the fix is NEW_PATTERN: add a prefix-slice-plus-cast materialization template for contiguous inputs that emits the direct load/cast/store traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _prefix_slice_bf16_to_f32_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, values, mask=mask)


# 0473776d: (T([50272,1024], bf16))
@oracle_impl(hardware="B200", point="0473776d", BLOCK=1024, num_warps=4)
# c7a73141: (T([50272,768], bf16))
@oracle_impl(hardware="B200", point="c7a73141", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (arg0_1,) = inputs
    rows = arg0_1.shape[0] - 7
    cols = arg0_1.shape[1]
    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    n_elements = rows * cols
    grid = (triton.cdiv(n_elements, BLOCK),)
    _prefix_slice_bf16_to_f32_kernel[grid](
        arg0_1,
        output,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return output
