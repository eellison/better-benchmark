"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle materializes the fp32 input into one contiguous bf16 buffer and returns both the original-layout tensor and its transposed metadata view from that single storage, whereas Inductor currently lowers this multi-output cast-plus-view region through generic output planning instead of treating the permute as a pure alias of the cast output; Inductor cannot do this today because its pointwise scheduler/output planner does not canonicalize returned post-cast view aliases into one materialized buffer with sibling metadata views; the fix is ALGEBRAIC_ELIMINATION: coalesce post-cast view returns during output planning and emit only the required cast materialization."""

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


@oracle_impl(hardware="B200", point="44a2434c", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ae9f1068", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f5f85987", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="774eab0e", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK=1024, num_warps=4):
    arg0_1 = inputs[0]
    rows, cols = arg0_1.shape
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
