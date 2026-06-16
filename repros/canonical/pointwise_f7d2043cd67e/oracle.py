"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Gemma bf16 `30 * tanh(x / 30)` pointwise scope as one storage-linear Triton kernel, including the metadata-only `[1000, 256000] -> [1, 1000, 256000]` view, fp32 natural `libdevice.tanh` evaluation, and final bf16 output rounding in the returned contiguous 3D layout, whereas Inductor lowers the same view/div/tanh/mul chain through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no B200-tuned dense unary transcendental template for this very large contiguous Gemma activation clamp and relies on generic scheduling choices; the fix is NEW_PATTERN: add a guarded bf16 tanh-scale pointwise specialization or equivalent autotuned launch choice for large contiguous transformer activations."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


TOTAL = 256000000


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _tanh_scale_kernel(
    x_ptr,
    out_ptr,
    TOTAL_: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL_
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = _div_rn_f32(x, 30.0)
    y = _mul_rn_f32(libdevice.tanh(scaled), 30.0)
    tl.store(out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="736b279f", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int):
    x, _shape = inputs
    del _shape
    out = torch.empty_strided(
        (1, 1000, 256000),
        (TOTAL, 256000, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _tanh_scale_kernel[(triton.cdiv(TOTAL, BLOCK_SIZE),)](
        x,
        out,
        TOTAL_=TOTAL,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
