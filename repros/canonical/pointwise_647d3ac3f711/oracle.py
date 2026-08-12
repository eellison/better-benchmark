"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 `full([tail], 0)` plus one-dimensional `cat([input, full])` scope by directly materializing the fresh contiguous cat output in one Triton kernel, copying the input prefix and writing the exact bf16 zero suffix, whereas Inductor lowers the decomposed full and cat through its generic pointwise/layout indexing path; Inductor cannot do this today because its scheduler does not sink a small constant-full producer into the cat layout store as a single shape-specialized output materialization; the fix is SCHEDULER_FUSION: teach cat scheduling to fuse constant full/empty stencil inputs into the final cat store while preserving the output allocation, dtype, and contiguous stride contract."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cat_suffix_zero_kernel(
    input_ptr,
    output_ptr,
    input_numel: tl.constexpr,
    output_numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    copy_mask = offsets < input_numel
    store_mask = offsets < output_numel
    values = tl.load(input_ptr + offsets, mask=copy_mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=store_mask)


@oracle_impl(hardware="B200", point="ff177488", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="bcb2f702", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="81752f72", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="0bd4da28", BLOCK=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, _shape_param_0 = inputs
    tail = int(_shape_param_0[0])
    input_numel = arg0_1.numel()
    output_numel = input_numel + tail

    output = torch.empty_strided(
        (output_numel,),
        (1,),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    grid = (triton.cdiv(output_numel, BLOCK),)
    _cat_suffix_zero_kernel[grid](
        arg0_1,
        output,
        input_numel=input_numel,
        output_numel=output_numel,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
