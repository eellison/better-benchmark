"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 VisFormer exact-GELU-derivative pointwise scope in one storage-linear Triton kernel, including bf16 input promotion to fp32, the exact `0.5*(erf(x*0.7071067811865476)+1) + x*exp(-0.5*x*x)*0.3989422804014327` derivative formulation, multiplication by the fp32 upstream gradient, final bf16 cast, and the captured channels-last output strides for all shape points, whereas Inductor lowers the same isolated expensive pointwise expression through its generic fused pointwise codegen; Inductor cannot do this today because pointwise codegen has no B200-tuned exact-GELU-derivative template for this channels-last VisFormer shape family; the fix is NEW_PATTERN: add a guarded exact-GELU-derivative pointwise specialization or equivalent autotuned launch choice when it beats generic pointwise codegen."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gelu_derivative_bf16_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    erf_arg = x * 0.7071067811865476
    erf_val = libdevice.erf(erf_arg)
    cdf = (erf_val + 1.0) * 0.5
    x2 = x * x
    exp_arg = x2 * -0.5
    exp_val = libdevice.exp(exp_arg)
    pdf = exp_val * 0.3989422804014327
    correction = x * pdf
    derivative = cdf + correction
    out = grad * derivative
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


# f961de61: (T([128,384,28,28], bf16, stride=(301056,1,10752,384)), ...)
@oracle_impl(hardware="B200", point="f961de61", BLOCK=512, num_warps=4)
# 21c46fcd: (T([128,1536,14,14], bf16, stride=(301056,1,21504,1536)), ...)
@oracle_impl(hardware="B200", point="21c46fcd", BLOCK=512, num_warps=4)
# cfcbfec7: (T([128,3072,7,7], bf16, stride=(150528,1,21504,3072)), ...)
@oracle_impl(hardware="B200", point="cfcbfec7", BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    arg0_1, arg1_1 = inputs
    out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_elements = arg0_1.numel()
    _gelu_derivative_bf16_kernel[(triton.cdiv(n_elements, BLOCK),)](
        arg0_1,
        arg1_1,
        out,
        n_elements=n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
