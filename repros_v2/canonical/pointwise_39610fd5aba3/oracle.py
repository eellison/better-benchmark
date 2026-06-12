"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle collapses the unsqueeze/view/squeeze chain into output metadata and materializes only the required contiguous f32-to-bf16 cast buffer with one storage-linear Triton kernel, whereas Inductor schedules the captured view-only chain through its generic pointwise dtype-conversion path for the logical reshaped result; Inductor cannot do this today because output planning does not fully cancel size-one unsqueeze/view/squeeze operations before selecting the cast schedule; the fix is ALGEBRAIC_ELIMINATION: canonicalize size-one view chains before pointwise scheduling so codegen emits the direct storage-linear cast into the final dense layout."""

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
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, shape_param = inputs
    rows = int(shape_param[1])
    cols = int(shape_param[2])
    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _f32_to_bf16_kernel[(triton.cdiv(arg0_1.numel(), BLOCK),)](
        arg0_1,
        output,
        n_elements=arg0_1.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return output
