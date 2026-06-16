"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle materializes the fp32 input into one contiguous bf16 buffer and returns both the squeezed view and its transposed metadata view from that single storage, whereas Inductor currently lowers the two returned post-cast layouts through generic output planning instead of treating the permute as a pure alias of the same cast buffer; Inductor cannot do this today because its pointwise scheduler/output planner does not canonicalize sibling returned views after canceling size-one unsqueeze/view/squeeze operations; the fix is ALGEBRAIC_ELIMINATION: coalesce post-cast view returns during output planning and emit only the required cast materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_to_bf16_kernel(input_ptr, output_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="d102a86e", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK=1024, num_warps=4):
    arg0_1, shape_param = inputs
    rows = int(shape_param[1])
    cols = int(shape_param[2])
    base = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(arg0_1.numel(), BLOCK),)
    _f32_to_bf16_kernel[grid](
        arg0_1,
        base,
        n_elements=arg0_1.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return base, base.permute(1, 0)
