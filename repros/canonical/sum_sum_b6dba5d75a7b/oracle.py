"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete tiny BERT softmax-backward tail in one Triton kernel, including the f32 row sum, natural f32 exp/multiply/subtract producer, bf16 dense output, returned permute alias, and bf16-roundtripped f32 column sum, whereas Inductor schedules the row reduction, producer/materialization, layout view, and sibling column reduction as separate generic regions; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output plan for this small cross-axis producer with a view output and a dependent post-cast reduction; the fix is SCHEDULER_FUSION: add a guarded multi-output fusion that keeps the row-local summary live, writes the materialized bf16 tensor once, returns its layout alias, and accumulates the compatible column sum from the same producer."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 16
COLS = 2
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)
PERMUTE_SHAPE = (COLS, ROWS)
PERMUTE_STRIDE = (1, COLS)


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
def _softmax_backward_tiny_kernel(
    logits_ptr,
    grad_ptr,
    dense_out_ptr,
    sum_out_ptr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    offsets0 = rows * 2
    offsets1 = offsets0 + 1
    active = rows < 16

    logit0 = tl.load(logits_ptr + offsets0, mask=active, other=0.0).to(tl.float32)
    logit1 = tl.load(logits_ptr + offsets1, mask=active, other=0.0).to(tl.float32)
    grad0 = tl.load(grad_ptr + offsets0, mask=active, other=0.0).to(tl.float32)
    grad1 = tl.load(grad_ptr + offsets1, mask=active, other=0.0).to(tl.float32)

    row_sum = _f32_add(grad0, grad1)
    value0 = _f32_sub(grad0, _f32_mul(libdevice.exp(logit0), row_sum))
    value1 = _f32_sub(grad1, _f32_mul(libdevice.exp(logit1), row_sum))
    dense0 = value0.to(tl.bfloat16, fp_downcast_rounding="rtne")
    dense1 = value1.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(dense_out_ptr + offsets0, dense0, mask=active)
    tl.store(dense_out_ptr + offsets1, dense1, mask=active)

    sum0 = tl.sum(tl.where(active, dense0.to(tl.float32), 0.0), axis=0)
    sum1 = tl.sum(tl.where(active, dense1.to(tl.float32), 0.0), axis=0)
    tl.store(sum_out_ptr + 0, sum0.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32))
    tl.store(sum_out_ptr + 1, sum1.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32))


@oracle_impl(hardware="B200", point="840e34c4", BLOCK_M=16, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    logits, grad, _shape = inputs
    dense_out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=grad.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((COLS,), (1,), device=grad.device, dtype=torch.float32)
    _softmax_backward_tiny_kernel[(1,)](
        logits,
        grad,
        dense_out,
        sum_out,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=3,
    )
    return dense_out, dense_out.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE), sum_out
