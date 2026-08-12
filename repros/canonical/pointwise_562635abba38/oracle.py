"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 ALBERT tanh-GELU backward pointwise scope in one storage-linear Triton kernel, including the metadata-only `[4096,16384] -> [8,512,16384] -> [4096,16384]` views, every captured bf16 cast/add boundary, the materialized bf16 return, and its `[16384,4096]` transpose alias, whereas Inductor lowers the decomposed view/cast/pow/tanh/cast/add/view/permute graph through generic pointwise and output-planning codegen; Inductor cannot do this today because its scheduler does not expose a guarded alias-aware multi-output pointwise template that shares the expensive tanh-GELU backward intermediates while preserving the bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach pointwise scheduling to generate this fused GELU-backward materialization with explicit dtype boundaries and metadata-only alias returns."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _albert_tanh_gelu_backward_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    half_x = (x * 0.5).to(tl.bfloat16).to(tl.float32)
    left = grad * half_x

    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    tanh_val = libdevice.tanh(tanh_arg)
    tanh_plus_one = tanh_val + 1.0

    right = grad * tanh_plus_one
    right_half = (right.to(tl.bfloat16).to(tl.float32) * 0.5).to(tl.bfloat16)

    tanh_grad = 1.0 - tanh_val * tanh_val
    d_tanh = (left * tanh_grad) * 0.7978845608028654
    d_tanh_bf16 = d_tanh.to(tl.bfloat16)
    d_cubic = ((d_tanh * 0.044715) * (x2 * 3.0)).to(tl.bfloat16)

    derivative_tail = (d_tanh_bf16.to(tl.float32) + d_cubic.to(tl.float32)).to(
        tl.bfloat16
    )
    out = (derivative_tail.to(tl.float32) + right_half.to(tl.float32)).to(
        tl.bfloat16
    )
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="271eb370", BLOCK_SIZE=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int):
    grad, x, _shape0, _shape1, shape2 = inputs
    out_shape = tuple(int(dim) for dim in shape2)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    _albert_tanh_gelu_backward_kernel[(triton.cdiv(grad.numel(), BLOCK_SIZE),)](
        grad,
        x,
        out,
        n_elements=grad.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out, out.permute(1, 0)
