"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 `aten.tanh.default` scope as one storage-linear Triton kernel with fp32 natural `libdevice.tanh` evaluation and bf16 output rounding for both captured contiguous shape points, whereas Inductor lowers the same isolated tanh through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no B200-tuned tiny/dense contiguous unary transcendental template beyond generic pointwise lowering; the fix is NEW_PATTERN: add a guarded bf16 tanh pointwise specialization or equivalent autotuned launch choice for isolated contiguous tanh patterns."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _tanh_bf16_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets).to(tl.float32)
    y = libdevice.tanh(x)
    tl.store(output_ptr + offsets, y)


# 3fee83c6: (T([1, 3, 256, 256], bf16))
@oracle_impl(hardware="B200", point="3fee83c6", BLOCK_SIZE=256, num_warps=8, num_stages=1)
# 5a025cf0: (T([128, 16], bf16))
@oracle_impl(hardware="B200", point="5a025cf0", BLOCK_SIZE=64, num_warps=4, num_stages=1)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int, num_stages: int):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _tanh_bf16_kernel[(triton.cdiv(x.numel(), BLOCK_SIZE),)](
        x,
        out,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
