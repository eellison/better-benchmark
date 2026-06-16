"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full bf16 exact-erf GELU scope, including the metadata-only `[M, N] -> [B, S, N] -> [M, N]` views, fp32 `0.5*x*(erf(0.7071067811865476*x)+1)` computation, bf16 rounding, and final contiguous output, in one storage-linear Triton kernel registered at every captured shape point, whereas Inductor lowers the same view-plus-GELU chain through its generic pointwise fusion path; Inductor cannot do this today because the scheduler has no dedicated bf16 exact-erf GELU layout template with B200-tuned launch parameters for this cross-model shape family; the fix is NEW_PATTERN: add a tuned exact-erf GELU pointwise template or equivalent autotuned specialization for metadata-view bf16 GELU patterns."""

import torch
import triton
import triton.language as tl
from oracle_harness import oracle_impl


@triton.jit
def _gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    half_x = x * 0.5
    erf_arg = x * 0.7071067811865476
    y = half_x * (tl.math.erf(erf_arg) + 1.0)
    tl.store(output_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="ddbf4fbd", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5666a344", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5ad79285", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5cb74639", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2079d386", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="a3b11b8d", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="de85fb74", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="d62a5c2a", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="6942db17", BLOCK_SIZE=512, num_warps=4)
@oracle_impl(hardware="B200", point="5d3406ab", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e73010dd", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e44d982c", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="92efc45e", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="c2111490", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="befcb921", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="50ec4979", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e223410f", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="67d3fea7", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="d87997ca", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2909fe19", BLOCK_SIZE=512, num_warps=4)
@oracle_impl(hardware="B200", point="1b310572", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    x, _shape0, shape1 = inputs
    output = torch.empty_strided(
        tuple(shape1),
        (int(shape1[1]), 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _gelu_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
