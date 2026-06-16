"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 SwiGLU pointwise scope in one storage-linear Triton kernel, including the metadata-only views, explicit fp32 cast of the first input, natural `exp(-x)`, add/div expression, bf16 rounding before the gate multiply, final bf16 output rounding, and returned contiguous `[1000,K]` layout at every shape point, whereas Inductor lowers the decomposed view/convert/neg/exp/add/div/convert/view/mul/view graph through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no B200-tuned dense bf16 SwiGLU template that specializes the view-only shape parameters and required low-precision rounding boundary; the fix is NEW_PATTERN: add a guarded bf16 SwiGLU pointwise template or equivalent pointwise autotuning specialization that preserves the explicit fp32 arithmetic and bf16 cast before the gate multiply."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _swiglu_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_MASK:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offsets).to(tl.float32)
        gate = tl.load(gate_ptr + offsets)

    activated = x / (libdevice.exp(-x) + 1.0)
    activated_bf16 = activated.to(tl.bfloat16)
    out = activated_bf16 * gate

    if USE_MASK:
        tl.store(out_ptr + offsets, out, mask=mask)
    else:
        tl.store(out_ptr + offsets, out)


# 22f70084: (T([1000,3072], bf16), T([1000,3072], bf16), S([1,1000,3072]), S([1,1000,3072]), S([1000,3072]))
# a1f43414: (T([1000,14336], bf16), T([1000,14336], bf16), S([1,1000,14336]), S([1,1000,14336]), S([1000,14336]))
@oracle_impl(hardware="B200", point="22f70084", BLOCK_SIZE=512, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="a1f43414", BLOCK_SIZE=2048, USE_MASK=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, USE_MASK, num_warps):
    x, gate, _shape0, _shape1, out_shape = inputs
    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    _swiglu_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x,
        gate,
        out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
    )
    return out
