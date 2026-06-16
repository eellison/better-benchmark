"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete TrOCR bf16 softmax-backward row update in one Triton kernel, including the `[1024,256,256] -> [64,16,256,256]` metadata view, fp32 broadcast add from the `[64,1,256,256]` tensor, fp32 subtract/`aten.exp`/divide by the row denominator, fp32 multiplication by the bf16 gradient input, the last-dimension `sum(..., keepdim=True)`, exact `prims.fma(-div, row_sum, mul)` epilogue via `fma.rn.f32`, bf16 conversion, and final contiguous `[1024,256,256]` view. Inductor already handles this as a generic fixed-width row reduction plus epilogue, so the local gap is mostly launch/memory/math floor rather than missing broader scope; the fix is BANDWIDTH_BOUND: record this as a full-scope floor check unless a broader row-reduction/softmax-backward template changes the baseline."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 1024 * 256
K = 256


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
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
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _softmax_backward_kernel(
    grad_ptr,
    logits_bf16_ptr,
    bias_ptr,
    row_shift_ptr,
    row_denom_ptr,
    out_ptr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_K)
    row_ids = rows[:, None]
    col_ids = cols[None, :]
    row_mask = rows < 262144
    mask = row_mask[:, None] & (col_ids < 256)

    offsets = row_ids * 256 + col_ids
    row0 = rows // 256
    row1 = rows - row0 * 256
    bias_batch = row0 // 16
    bias_offsets = bias_batch[:, None] * (256 * 256) + row1[:, None] * 256 + col_ids

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    logits = tl.load(logits_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)

    shifted = _sub_rn_f32(_add_rn_f32(logits, bias), shift[:, None])
    prob = _div_rn_f32(libdevice.exp(shifted), denom[:, None])
    product = _mul_rn_f32(grad, prob)
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)
    out = _fma_rn_f32(-prob, row_sum[:, None], product)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


# b0ff1d38: TrOCR bf16 softmax-backward row update, K=256.
@oracle_impl(hardware="B200", point="b0ff1d38", ROW_BLOCK=2, BLOCK_K=256, num_warps=4, num_stages=4)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_K: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    out = torch.empty_strided(
        (1024, 256, 256),
        (256 * 256, 256, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _softmax_backward_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
