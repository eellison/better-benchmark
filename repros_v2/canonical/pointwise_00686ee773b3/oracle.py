"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete reshape plus bf16-to-fp32 materialization scope as one storage-linear Triton cast over the contiguous input backing storage, returning the eager-compatible contiguous fp32 3D output shape and stride for every captured shape parameter, whereas Inductor lowers the metadata-only view feeding `prims.convert_element_type` through its generic pointwise materialization path; Inductor cannot do this today because its pointwise scheduler has no guarded pure dtype-conversion template that erases the reshape indexing scaffold and emits a direct contiguous widening copy for this HuggingFace hidden-state family; the fix is NEW_PATTERN: add a reshape-aware dtype-conversion materialization lowering that recognizes compact contiguous views and emits a direct storage-order cast plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bf16_to_f32_nomask_kernel(
    input_ptr,
    output_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets).to(tl.float32)
    tl.store(output_ptr + offsets, values)


@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK=8192, num_warps=8)
@oracle_impl(hardware="B200", point="90358d5b", BLOCK=8192, num_warps=8)
@oracle_impl(hardware="B200", point="bd432928", BLOCK=8192, num_warps=8)
@oracle_impl(hardware="B200", point="07bfd41e", BLOCK=8192, num_warps=8)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK=8192, num_warps=8)
@oracle_impl(hardware="B200", point="b642f4d6", BLOCK=8192, num_warps=8)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, shape_param = inputs
    dim0 = int(shape_param[0])
    dim1 = int(shape_param[1])
    dim2 = int(shape_param[2])
    output = torch.empty_strided(
        (dim0, dim1, dim2),
        (dim1 * dim2, dim2, 1),
        device=x.device,
        dtype=torch.float32,
    )
    n_elements = x.numel()
    _bf16_to_f32_nomask_kernel[(triton.cdiv(n_elements, BLOCK),)](
        x,
        output,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return output
