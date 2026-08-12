"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 NFNet SiLU-plus-identity-multiply pointwise scope as one storage-linear Triton kernel, including bf16 input promotion, natural-exp fp32 `x / (exp(-x) + 1)` arithmetic, explicit bf16 rounding, the captured `* 1.0` bf16 output operation, and the dense channels-last output layout for both points, whereas Inductor lowers the decomposed convert/neg/exp/add/div/convert/mul graph through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no B200-tuned bf16 SiLU materialization template that also preserves the post-cast identity multiply boundary; the fix is NEW_PATTERN: add a guarded bf16 SiLU pointwise lowering or equivalent autotuned specialization that preserves explicit dtype boundaries and folds only provably identity post-cast work."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _silu_mul1_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets).to(tl.float32)
    silu = x / (libdevice.exp(-x) + 1.0)
    rounded = silu.to(tl.bfloat16).to(tl.float32)
    out = rounded * 1.0
    tl.store(output_ptr + offsets, out)


# 34916808: T([128,128,72,72], bf16, stride=(663552,1,9216,128))
@oracle_impl(hardware="B200", point="34916808", BLOCK_SIZE=1024, num_warps=4)
# 73d37b7b: T([128,128,56,56], bf16, stride=(401408,1,7168,128))
@oracle_impl(hardware="B200", point="73d37b7b", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    _silu_mul1_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x,
        out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
