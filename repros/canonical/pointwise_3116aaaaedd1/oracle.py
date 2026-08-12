"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MT5 gated tanh-GELU backward pointwise scope in one storage-linear Triton kernel, including dropout-mask scaling, the shared natural-tanh GELU producer, both bf16 materialized outputs, and their metadata-only transpose aliases, whereas Inductor lowers the decomposed multi-output pointwise graph through generic pointwise/output-planning codegen; Inductor cannot do this today because its scheduler does not expose a guarded alias-aware multi-output pointwise template that shares the expensive tanh-GELU intermediates while preserving every bf16 rounding boundary; the fix is SCHEDULER_FUSION: teach pointwise scheduling to keep sibling GELU-backward outputs in one fused plan with explicit bf16 cast/add boundaries and alias-view output returns."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _mt5_gated_tanh_gelu_backward_kernel(
    grad_ptr,
    mask_ptr,
    x_ptr,
    gate_ptr,
    out0_ptr,
    out1_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
    keep = tl.load(mask_ptr + offsets, mask=in_bounds, other=0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)

    scaled_grad = grad * (keep * 1.1111111111111112)
    half = (x * 0.5).to(tl.bfloat16).to(tl.float32)

    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    tanh_val = libdevice.tanh(tanh_arg)
    tanh_plus_one = tanh_val + 1.0

    gelu = half * tanh_plus_one
    tl.store(out0_ptr + offsets, scaled_grad * gelu, mask=in_bounds)

    gated_grad = scaled_grad * gate
    left = gated_grad * half
    right = gated_grad * tanh_plus_one
    right_half = (right.to(tl.bfloat16).to(tl.float32) * 0.5).to(tl.bfloat16)

    tanh_grad = 1.0 - tanh_val * tanh_val
    d_tanh = ((left * tanh_grad) * 0.7978845608028654)
    d_tanh_bf16 = d_tanh.to(tl.bfloat16)
    d_cubic = ((d_tanh * 0.044715) * (x2 * 3.0)).to(tl.bfloat16)

    derivative_tail = (d_tanh_bf16.to(tl.float32) + d_cubic.to(tl.float32)).to(
        tl.bfloat16
    )
    out1 = (derivative_tail.to(tl.float32) + right_half.to(tl.float32)).to(
        tl.bfloat16
    )
    tl.store(out1_ptr + offsets, out1, mask=in_bounds)


@oracle_impl(hardware="B200", point="f749e533", BLOCK_SIZE=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    arg0, arg1, arg2, arg3, _shape0, _shape1, _shape2, out0_shape, out1_shape = inputs
    out0_shape = tuple(int(dim) for dim in out0_shape)
    out1_shape = tuple(int(dim) for dim in out1_shape)
    out0 = torch.empty_strided(
        out0_shape,
        (out0_shape[1], 1),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    out1 = torch.empty_strided(
        out1_shape,
        (out1_shape[1], 1),
        device=arg0.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(arg0.numel(), BLOCK_SIZE),)
    _mt5_gated_tanh_gelu_backward_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        out0,
        out1,
        n_elements=arg0.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out0, out0.permute(1, 0), out1, out1.permute(1, 0)
