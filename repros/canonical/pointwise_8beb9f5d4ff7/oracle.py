"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle collapses the unsqueeze/view/squeeze chain into metadata and materializes only the required contiguous bf16-to-float32 output with one storage-linear Triton cast, whereas Inductor currently schedules the captured reshape-only chain through generic pointwise dtype-conversion code for the logical view shape; Inductor cannot do this today because output planning does not fully cancel size-one view operations before selecting the pointwise cast indexing strategy; the fix is ALGEBRAIC_ELIMINATION: canonicalize size-one unsqueeze/view/squeeze chains before scheduling dtype conversions so codegen emits the direct storage-linear cast."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bf16_to_f32_kernel(in_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="a9384bfb", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK=1024, num_warps=4):
    arg0_1, shape_param = inputs
    out_shape = (shape_param[0], shape_param[1], shape_param[2])
    out = torch.empty_strided(
        out_shape,
        (shape_param[1] * shape_param[2], shape_param[2], 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    _bf16_to_f32_kernel[(triton.cdiv(arg0_1.numel(), BLOCK),)](
        arg0_1,
        out,
        n_elements=arg0_1.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
