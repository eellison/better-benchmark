"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 gated tanh-approximate GELU scope as one storage-linear Triton pointwise kernel, including the metadata-only views, explicit bf16-to-fp32 GELU arithmetic, bf16 rounding before the second-input multiply, and final contiguous bf16 output view, whereas Inductor lowers the same cast/GELU/cast/gate chain through generic pointwise fusion; Inductor cannot do this today because pointwise codegen has no guarded B200-tuned template for this transformer hidden-size family that preserves the mixed bf16/fp32 rounding boundaries and post-cast gate multiply; the fix is NEW_PATTERN: add a dedicated gated tanh-GELU pointwise template or equivalent autotuned specialization for metadata-view transformer activation gates."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gated_tanh_gelu_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets).to(tl.float32)
    gate = tl.load(gate_ptr + offsets)
    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    gelu = (x * 0.5) * (libdevice.tanh(tanh_arg) + 1.0)
    y = gelu.to(tl.bfloat16) * gate
    tl.store(out_ptr + offsets, y)


@oracle_impl(hardware="B200", point="4c39a052", BLOCK_SIZE=1024, num_warps=8)
@oracle_impl(hardware="B200", point="5e420cbf", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    x, gate, _shape0, _shape1, _shape2 = inputs
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    _gated_tanh_gelu_kernel[grid](
        x,
        gate,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
