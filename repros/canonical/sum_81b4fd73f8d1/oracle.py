"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 BART-family cross-entropy-backward dual-pad tail, including scalar f32 division, ignore-index label handling, the sparse one-hot gradient's bf16 round trip before its row sum, natural `exp` after the bf16-rounded `logit - row_shift0 - row_shift1`, bf16 correction rounding, bf16 residual add, returned transposed padded `[50272, rows]` output, and returned row-major padded `[rows, 50272]` output, while replacing the materialized dense one-hot tensor and its row reduction with the equivalent guarded per-row label scalar; Inductor lowers the captured div/iota/eq/where/mul/sum/exp/add/permute/constant_pad graph as generic one-hot materialization, reduction, dense pointwise, and two layout stores; Inductor cannot do this today because algebraic simplification does not canonicalize equality-built one-hot reductions into guarded label-indexed row scalars while preserving the bf16/f32 rounding boundaries and dual padded output contract; the fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before scheduling and emit the dense bf16 epilogue directly into both padded layouts."""

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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _dual_pad_ce_backward_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    out_t_ptr,
    out_row_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    PADDED_N: tl.constexpr,
    LOGITS_ROW_STRIDE_N: tl.constexpr,
    RESIDUAL_ROW_STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS_N
    padded_col_mask = cols < PADDED_N
    data_col_mask = cols < VOCAB_N
    data_mask = row_mask[:, None] & data_col_mask[None, :]
    out_mask = row_mask[:, None] & padded_col_mask[None, :]

    scale = _f32_div(
        tl.load(numerator_ptr).to(tl.float32),
        tl.load(denominator_ptr).to(tl.float32),
    )
    label = tl.load(labels_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
    active = label != -100
    safe_label = tl.where(active, label, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)

    zero = tl.full((), 0.0, tl.float32)
    row_scale = tl.where(active, scale, zero)
    one_hot = tl.where(safe_label[:, None] == cols[None, :], -1.0, 0.0)
    one_hot_scaled = _f32_mul(one_hot, row_scale[:, None])
    label_grad = one_hot_scaled.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == zero
    rounded_neg_scale = _f32_mul(-1.0, scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    finite_row_sum = tl.where(in_vocab, rounded_neg_scale, zero)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero)

    logits_offsets = rows[:, None] * LOGITS_ROW_STRIDE_N + cols[None, :]
    residual_offsets = rows[:, None] * RESIDUAL_ROW_STRIDE_N + cols[None, :]
    logits = tl.load(
        logits_ptr + logits_offsets,
        mask=data_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    shifted = _f32_sub(logits, row_shift0[:, None])
    shifted = _f32_sub(shifted, row_shift1[:, None])
    shifted = shifted.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    exp_shifted = libdevice.exp(shifted)

    correction = _f32_sub(label_grad, _f32_mul(exp_shifted, row_sum[:, None])).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    residual = tl.load(
        residual_ptr + residual_offsets,
        mask=data_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    value = _bf16_add(residual, correction)
    value = tl.where(data_mask, value, zero.to(tl.bfloat16))

    tl.store(out_row_ptr + rows[:, None] * PADDED_N + cols[None, :], value, mask=out_mask)
    tl.store(out_t_ptr + cols[None, :] * ROWS_N + rows[:, None], value, mask=out_mask)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        _shape0,
        shape1,
        _shape2,
        _shape3,
        _shape4,
        _shape5,
        _shape6,
    ) = inputs

    rows = int(labels.numel())
    vocab = int(shape1[1])
    padded = int(logits.stride(-2))
    out_t = torch.empty_strided(
        (padded, rows),
        (rows, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    out_row = torch.empty_strided(
        (rows, padded),
        (padded, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )

    _dual_pad_ce_backward_kernel[
        (triton.cdiv(padded, BLOCK_N), triton.cdiv(rows, BLOCK_M))
    ](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_t,
        out_row,
        ROWS_N=rows,
        VOCAB_N=vocab,
        PADDED_N=padded,
        LOGITS_ROW_STRIDE_N=logits.stride(-2),
        RESIDUAL_ROW_STRIDE_N=residual.stride(-2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out_t, out_row


# 5599a41a: Bart/MBart train, rows=8192, vocab=50265, padded=50272.
@oracle_impl(hardware="B200", point="5599a41a", BLOCK_M=16, BLOCK_N=256, num_warps=8)
# 2167944f: Pegasus train, rows=16384, vocab=50265, padded=50272.
@oracle_impl(hardware="B200", point="2167944f", BLOCK_M=16, BLOCK_N=256, num_warps=8)
# 393d4aa1: TrOCR train, rows=16384, vocab=50265, padded=50272.
@oracle_impl(hardware="B200", point="393d4aa1", BLOCK_M=16, BLOCK_N=256, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps)
