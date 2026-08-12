"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GPT-Neo sequence-classification training scatter tail in one Triton materialization kernel, including the scalar div/where producer, bf16 round-trip of the upstream mask gradient, two-class stable log-softmax with the explicit bf16 log-probability round before exponentiation, bf16 residual add, duplicate-preserving index_put(accumulate=True) into the logical [32,128,2] buffer, and returned contiguous view plus transposed alias, whereas Inductor lowers the softmax-gradient producer, zero fill, generic scatter-add, view, and permute as separate scheduled work around a materialized scatter buffer; Inductor cannot do this today because its scheduler/codegen does not recognize a row-local softmax-backward producer feeding a tiny structured duplicate-index scatter-reduce whose live outputs are only aliases of the scatter storage; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that keeps the row-local producer in registers and materializes the required aliasing output layout directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _zero_view_kernel(
    scalar_out_ptr,
    view_out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL

    tl.store(scalar_out_ptr, 0.0, mask=pid == 0)
    tl.store(view_out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32).to(tl.bfloat16), mask=active)


@triton.jit
def _scatter_accum_kernel(
    arg0_ptr,
    arg1_ptr,
    mask_ptr,
    row_mask_ptr,
    logits_ptr,
    residual_ptr,
    index0_ptr,
    index1_ptr,
    view_out_ptr,
    ROWS: tl.constexpr,
    SCATTER_COLS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.arange(0, ROW_BLOCK)
    active = rows < ROWS

    div = _f32_div(tl.load(arg0_ptr).to(tl.float32), tl.load(arg1_ptr).to(tl.float32))

    row_enabled = tl.load(row_mask_ptr + rows, mask=active, other=0)
    scale = tl.where(row_enabled, div, 0.0)

    mask0 = tl.load(mask_ptr + rows * 2, mask=active, other=0)
    mask1 = tl.load(mask_ptr + rows * 2 + 1, mask=active, other=0)
    upstream0 = _f32_mul(tl.where(mask0, -1.0, 0.0), scale).to(tl.bfloat16).to(tl.float32)
    upstream1 = _f32_mul(tl.where(mask1, -1.0, 0.0), scale).to(tl.bfloat16).to(tl.float32)
    upstream_sum = _f32_add(upstream0, upstream1)

    logit0 = tl.load(logits_ptr + rows * 2, mask=active, other=0.0).to(tl.float32)
    logit1 = tl.load(logits_ptr + rows * 2 + 1, mask=active, other=0.0).to(tl.float32)
    row_max = tl.maximum(logit0, logit1)
    shifted0 = _f32_sub(logit0, row_max)
    shifted1 = _f32_sub(logit1, row_max)
    denom = _f32_add(libdevice.exp(shifted0), libdevice.exp(shifted1))
    log_denom = libdevice.log(denom)
    log_prob0 = _f32_sub(shifted0, log_denom).to(tl.bfloat16).to(tl.float32)
    log_prob1 = _f32_sub(shifted1, log_denom).to(tl.bfloat16).to(tl.float32)
    prob0 = libdevice.exp(log_prob0)
    prob1 = libdevice.exp(log_prob1)

    grad0 = _f32_sub(upstream0, _f32_mul(prob0, upstream_sum)).to(tl.bfloat16).to(tl.float32)
    grad1 = _f32_sub(upstream1, _f32_mul(prob1, upstream_sum)).to(tl.bfloat16).to(tl.float32)
    out0 = _f32_add(tl.load(residual_ptr + rows * 2, mask=active, other=0.0).to(tl.float32), grad0).to(tl.bfloat16)
    out1 = _f32_add(tl.load(residual_ptr + rows * 2 + 1, mask=active, other=0.0).to(tl.float32), grad1).to(tl.bfloat16)

    dst_row = tl.load(index0_ptr + rows, mask=active, other=0).to(tl.int64)
    dst_col = tl.load(index1_ptr + rows, mask=active, other=0).to(tl.int64)
    base = (dst_row * SCATTER_COLS + dst_col) * 2
    tl.atomic_add(view_out_ptr + base, out0, sem="relaxed", mask=active)
    tl.atomic_add(view_out_ptr + base + 1, out1, sem="relaxed", mask=active)


# 535c8dce: (T([], f32), T([], f32), T([32,2], b8), T([32,1], b8), T([32,2], bf16), T([32,2], bf16), T([32], i64, gen=Index(32)), T([32], i64, gen=Index(32)), S([32,128,2]), S([4096,2]))
@oracle_impl(
    hardware="B200",
    point="535c8dce",
    ZERO_BLOCK=1024,
    ROW_BLOCK=32,
    num_warps_zero=4,
    num_warps_scatter=1,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    ZERO_BLOCK: int,
    ROW_BLOCK: int,
    num_warps_zero: int,
    num_warps_scatter: int,
    num_stages: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        _shape_param_0,
        _shape_param_1,
    ) = inputs

    view_shape = tuple(int(dim) for dim in _shape_param_1)
    full = torch.empty((), device=arg4_1.device, dtype=torch.float32)
    view = torch.empty_strided(view_shape, (2, 1), device=arg4_1.device, dtype=torch.bfloat16)

    rows = int(arg2_1.shape[0])
    scatter_cols = int(_shape_param_0[1])
    total = int(view.numel())
    _zero_view_kernel[(triton.cdiv(total, ZERO_BLOCK),)](
        full,
        view,
        TOTAL=total,
        BLOCK=ZERO_BLOCK,
        num_warps=num_warps_zero,
        num_stages=num_stages,
    )
    _scatter_accum_kernel[(1,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        view,
        ROWS=rows,
        SCATTER_COLS=scatter_cols,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps_scatter,
        num_stages=num_stages,
    )
    return full, view, view.permute(1, 0)
