"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MobileViT SiLU pointwise scope with one storage-linear Triton kernel, including the metadata-only `[N,K] -> [512,*,K] -> [N,K]` views, explicit bf16-to-fp32 promotion, natural `exp(-x)`, add/div expression, and final bf16 output rounding for every captured point, whereas Inductor lowers the decomposed view/convert/neg/exp/add/div/convert/view graph through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no B200-tuned dense bf16 SiLU template that specializes the MobileViT view-only shape family while preserving the explicit fp32 arithmetic and bf16 cast boundary; the fix is NEW_PATTERN: add a guarded bf16 SiLU pointwise materialization template or equivalent pointwise autotuning specialization for these dense shapes."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _silu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_MASK:
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(input_ptr + offsets).to(tl.float32)

    y = x / (libdevice.exp(-x) + 1.0)

    if USE_MASK:
        tl.store(output_ptr + offsets, y, mask=mask)
    else:
        tl.store(output_ptr + offsets, y)


@oracle_impl(hardware="B200", point="fd0f60cd", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="4bf70c65", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="7f1f1e95", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, USE_MASK, num_warps):
    input_tensor, _shape0, output_shape = inputs
    output_shape = tuple(int(dim) for dim in output_shape)
    out = torch.empty_strided(
        output_shape,
        (output_shape[1], 1),
        device=input_tensor.device,
        dtype=torch.bfloat16,
    )
    n_elements = input_tensor.numel()
    _silu_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        input_tensor,
        out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
    )
    return out
