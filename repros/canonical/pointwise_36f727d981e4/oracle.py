"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle collapses the metadata-only `[B, S, H] -> [B*S, H]` view after the fp32-to-bf16 cast into output planning and materializes only the required contiguous bf16 buffer with one storage-linear Triton kernel for all captured transformer hidden-state points, whereas Inductor lowers the cast/view through its generic pointwise dtype-conversion path for the logical reshaped result; Inductor cannot do this today because output planning does not fully canonicalize a post-cast contiguous view before selecting the B200 cast schedule; the fix is ALGEBRAIC_ELIMINATION: canonicalize cast-followed-by-contiguous-view outputs so codegen emits a direct flat conversion into the final dense 2D layout."""

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


@oracle_impl(hardware="B200", point="b85aeb78", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="71639761", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="bcf9edde", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="784a7239", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="400995f1", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="139b073e", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="df1f991c", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="7997f4ec", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="0a855bca", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="727fdfe8", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK=1024, num_warps=4):
    arg0_1, shape_param = inputs
    rows = int(shape_param[0])
    cols = int(shape_param[1])
    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(arg0_1.numel(), BLOCK),)
    _f32_to_bf16_kernel[grid](
        arg0_1,
        output,
        n_elements=arg0_1.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return output
