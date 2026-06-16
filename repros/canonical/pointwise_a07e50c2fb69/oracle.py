"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full MT5 bf16 tanh-approximate GELU times the second dense input in one storage-linear kernel, including the metadata-only views, per-op bf16 rounding boundaries, natural tanh, and final contiguous `[4096, 1024]` output; whereas Inductor lowers the same decomposed view and pointwise expression through its generic flattened pointwise scheduler; Inductor cannot do this today because its pointwise codegen has no shape-specialized GELU-gated multiply template that preserves bf16 scalar-operation rounding while minimizing generic indexing overhead; the fix is NEW_PATTERN: add a guarded bf16 tanh-GELU multiply pointwise template for contiguous transformer MLP gates."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


TOTAL = 4096 * 1024


@triton.jit
def _bf16_gelu_mul_kernel(
    x_ptr,
    rhs_ptr,
    out_ptr,
    TOTAL_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL_ELEMENTS
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    half_x = (x * 0.5).to(tl.bfloat16).to(tl.float32)
    x2 = (x * x).to(tl.bfloat16).to(tl.float32)
    x3 = (x2 * x).to(tl.bfloat16).to(tl.float32)
    cubic = (x3 * 0.044715).to(tl.bfloat16).to(tl.float32)
    inner = (x + cubic).to(tl.bfloat16).to(tl.float32)
    tanh_arg = (inner * 0.7978845608028654).to(tl.bfloat16).to(tl.float32)
    tanh_val = libdevice.tanh(tanh_arg).to(tl.bfloat16).to(tl.float32)
    tanh_plus_one = (tanh_val + 1.0).to(tl.bfloat16).to(tl.float32)
    gelu = (half_x * tanh_plus_one).to(tl.bfloat16).to(tl.float32)
    out = (gelu * rhs).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


def _launch(inputs, *, BLOCK, num_warps):
    x, rhs, _shape0, _shape1, _shape2 = inputs
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=x.dtype)
    _bf16_gelu_mul_kernel[(triton.cdiv(TOTAL, BLOCK),)](
        x,
        rhs,
        out,
        TOTAL_ELEMENTS=TOTAL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out


@oracle_impl(hardware="B200", point="4ff4f886", BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    return _launch(inputs, BLOCK=BLOCK, num_warps=num_warps)
