"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GPT-J shifted-label bf16 cross-entropy backward tail, including ignore-index handling, scalar f32 division, natural `exp(logit - row_shift0 - row_shift1)`, the bf16 correction cast and residual add, the returned dense `[128, 50400]` buffer, its transposed alias, and the returned bf16-rounded f32 vocabulary sum, whereas Inductor materializes the expanded one-hot producer, reduces it to recover a per-row scalar, applies the dense exp epilogue and bf16 add, then separately reduces the final dense buffer; Inductor cannot do this today because its reduction canonicalization does not prove that the one-hot masked row reduction is just the guarded label-indexed scale, so the scheduler misses the full multi-output materialize-and-reduce fusion; the fix is ALGEBRAIC_ELIMINATION: canonicalize one-hot masked reductions into label-indexed scalar formulas and sink the derived scalar into a fused multi-output reduction template that emits all returned outputs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 128
VOCAB = 50400


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _backward_and_sum_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    out_base_ptr,
    out_sum_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_block = tl.program_id(0)

    rows = tl.arange(0, ROWS_N)[:, None]
    cols_1d = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = cols_1d[None, :]
    col_mask = cols < VOCAB_N
    offsets = rows * VOCAB_N + cols

    raw_label = tl.load(labels_ptr + 1 + rows).to(tl.int64)
    active = raw_label != -100
    safe_label = tl.where(active, raw_label, 0)

    numerator = tl.load(numerator_ptr).to(tl.float32)
    denominator = tl.load(denominator_ptr).to(tl.float32)
    scale = numerator / denominator

    zero_f32 = tl.full((), 0.0, tl.float32)
    neg_one_f32 = tl.full((), -1.0, tl.float32)
    row_scale = tl.where(active, scale, zero_f32)

    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)
    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == zero_f32
    finite_row_sum = tl.where(in_vocab, _f32_mul(neg_one_f32, row_scale), zero_f32)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero_f32)

    one_hot = tl.where(safe_label == cols, neg_one_f32, zero_f32)
    one_hot_scaled = _f32_mul(one_hot, row_scale)

    logits = tl.load(logits_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    row_shift0 = tl.load(row_shift0_ptr + rows).to(tl.float32)
    row_shift1 = tl.load(row_shift1_ptr + rows).to(tl.float32)
    centered = _f32_sub(logits, row_shift0)
    centered = _f32_sub(centered, row_shift1)
    exp_values = libdevice.exp(centered)

    correction = _f32_sub(one_hot_scaled, _f32_mul(exp_values, row_sum))
    correction_bf16 = correction.to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.load(residual_ptr + offsets, mask=col_mask, other=0.0)
    out_value = _bf16_add(residual, correction_bf16)

    tl.store(out_base_ptr + offsets, out_value, mask=col_mask)

    col_sum = tl.sum(out_value.to(tl.float32), axis=0)
    rounded_sum = col_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_sum_ptr + cols_1d, rounded_sum, mask=cols_1d < VOCAB_N)


@oracle_impl(
    hardware="B200",
    point="4e3e4a7e",
    BLOCK_N=64,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        _shape0,
        _shape1,
        _shape2,
        _shape3,
        _shape4,
        _shape5,
    ) = inputs

    out_base = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    out_sum = torch.empty_strided((VOCAB,), (1,), device=logits.device, dtype=torch.float32)

    _backward_and_sum_kernel[(triton.cdiv(VOCAB, BLOCK_N),)](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_base,
        out_sum,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )

    return out_base, out_base.permute(1, 0), out_sum
