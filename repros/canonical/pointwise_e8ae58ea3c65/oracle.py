"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 DCGAN sigmoid-backward pointwise scope in one storage-linear Triton kernel, including both bf16 input promotions to fp32, the captured `1 - sigmoid` subtraction, left-associated fp32 multiply `sigmoid * (1 - sigmoid)`, fp32 multiply by the upstream gradient, and final bf16 output rounding for the fresh `[32,1,1,1]` tensor, whereas Inductor already lowers this isolated 32-element expression through one generic fused pointwise kernel; Inductor cannot materially improve this local repro with scheduler fusion, algebraic elimination, split-K, scatter-reduce, or a new stencil pattern because there is no adjacent producer/consumer and the remaining work is the mandatory tiny allocation, launch, two reads, arithmetic, and one write; the fix is BANDWIDTH_BOUND: record this as a tiny pointwise launch/materialization floor unless broader pointwise launch overhead changes move both paths."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sigmoid_backward_bf16_kernel(
    grad_ptr,
    sigmoid_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.load(sigmoid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    one_minus = _f32_sub(1.0, sigmoid)
    partial = _f32_mul(sigmoid, one_minus)
    out = _f32_mul(grad, partial)
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="8ebdd403", BLOCK=64, num_warps=1)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    grad, sigmoid = inputs
    out = torch.empty_strided(
        tuple(grad.shape),
        tuple(grad.stride()),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    n_elements = grad.numel()
    grid = (triton.cdiv(n_elements, BLOCK),)
    _sigmoid_backward_bf16_kernel[grid](
        grad,
        sigmoid,
        out,
        n_elements=n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
