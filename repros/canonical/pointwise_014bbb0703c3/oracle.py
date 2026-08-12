"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 exact-erf GELU scope in one storage-linear Triton kernel, including dense channels-last or contiguous input layout, fp32 `0.5*x*(erf(0.7071067811865476*x)+1)` computation, explicit bf16 output rounding, and captured output strides for every shape point, whereas Inductor lowers the same isolated activation through its generic fused pointwise path; Inductor cannot do this today because pointwise codegen has no dedicated B200-tuned exact-erf GELU template for this channels-last/sequence shape family; the fix is NEW_PATTERN: add a guarded exact-erf GELU pointwise specialization or equivalent autotuned launch choice when it beats generic pointwise codegen."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _gelu_bf16_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    half_x = x * 0.5
    erf_arg = x * 0.7071067811865476
    y = half_x * (tl.math.erf(erf_arg) + 1.0)
    tl.store(output_ptr + offsets, y, mask=mask)


# ff8c3966: (T([128,3072,7,7], bf16, stride=(150528,1,21504,3072)))
@oracle_impl(hardware="B200", point="ff8c3966", BLOCK=1024, num_warps=4)
# d3381898: (T([128,1536,14,14], bf16, stride=(301056,1,21504,1536)))
@oracle_impl(hardware="B200", point="d3381898", BLOCK=1024, num_warps=4)
# e421f3e7: (T([128,384,28,28], bf16, stride=(301056,1,10752,384)))
@oracle_impl(hardware="B200", point="e421f3e7", BLOCK=1024, num_warps=4)
# af87e8bd: (T([1,384,3000], bf16))
@oracle_impl(hardware="B200", point="af87e8bd", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK),)
    _gelu_bf16_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
