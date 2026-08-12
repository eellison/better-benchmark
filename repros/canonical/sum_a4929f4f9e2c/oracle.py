"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Demucs bf16 GLU-derivative cat plus channel-sum scope in one producer/reduction Triton schedule, reusing each f32 sigmoid over the paired channel halves, materializing the returned contiguous bf16 `[4,4096,T]` tensor, and accumulating the returned f32 `[4096]` vector from those same bf16-rounded values, whereas Inductor lowers the cat producer and sibling `[0,2]` reduction through generic materialization and reduction scheduling; Inductor cannot do this today because its reduction codegen does not fuse a visible concatenated low-precision producer with its dependent channel reduction while preserving the explicit bf16 cast before the sum; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to emit a fused materialize-plus-channel-reduction producer for returned tensors with sibling reductions and shared sigmoid work."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 2048
C_TOTAL = 4096
BLOCK_K = 512


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cat_sum_kernel(
    grad_ptr,
    gate_pair_ptr,
    cat_out_ptr,
    sum_out_ptr,
    TIME: tl.constexpr,
    BATCH_: tl.constexpr,
    C_HALF_: tl.constexpr,
    C_TOTAL_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K_)
    k_total: tl.constexpr = BATCH_ * TIME
    mask = k < k_total
    b = k // TIME
    t = k - b * TIME

    grad_offsets = b * (C_HALF_ * TIME) + c * TIME + t
    pair_base = b * (C_TOTAL_ * TIME) + c * TIME + t
    cat_base = b * (C_TOTAL_ * TIME) + c * TIME + t

    grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    first_half = tl.load(gate_pair_ptr + pair_base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_pair_ptr + pair_base + C_HALF_ * TIME, mask=mask, other=0.0).to(tl.float32)

    sigmoid = _f32_div(1.0, _f32_add(libdevice.exp(_f32_sub(0.0, gate)), 1.0))
    first = _f32_mul(sigmoid, grad)
    second = _f32_mul(_f32_mul(_f32_mul(_f32_sub(1.0, sigmoid), sigmoid), first_half), grad)
    first_bf16 = first.to(tl.bfloat16)
    second_bf16 = second.to(tl.bfloat16)

    tl.store(cat_out_ptr + cat_base, first_bf16, mask=mask)
    tl.store(cat_out_ptr + cat_base + C_HALF_ * TIME, second_bf16, mask=mask)

    first_sum = tl.sum(tl.where(mask, first_bf16.to(tl.float32), 0.0), axis=0)
    second_sum = tl.sum(tl.where(mask, second_bf16.to(tl.float32), 0.0), axis=0)
    tl.store(sum_out_ptr + c, first_sum.to(tl.bfloat16).to(tl.float32))
    tl.store(sum_out_ptr + c + C_HALF_, second_sum.to(tl.bfloat16).to(tl.float32))


@oracle_impl(hardware="B200", point="61711b7c", TIME=90, num_warps=8)
@oracle_impl(hardware="B200", point="e6148dfd", TIME=92, num_warps=8)
def oracle_forward(inputs, *, TIME: int, num_warps: int):
    grad, gate_pair = inputs
    cat_out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        (C_TOTAL,),
        (1,),
        device=grad.device,
        dtype=torch.float32,
    )
    _cat_sum_kernel[(C_HALF,)](
        grad,
        gate_pair,
        cat_out,
        sum_out,
        TIME=TIME,
        BATCH_=BATCH,
        C_HALF_=C_HALF,
        C_TOTAL_=C_TOTAL,
        BLOCK_K_=BLOCK_K,
        num_warps=num_warps,
    )
    return cat_out, sum_out
