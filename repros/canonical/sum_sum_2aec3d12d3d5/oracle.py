"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Roberta causal-LM bf16 loss-backward fragment, including the shifted label slice, ignore-index handling, scalar f32 division, natural `exp((logit - row_shift0) - row_shift1)`, the bf16 correction cast and residual add, both returned padded dense layouts, and the returned bf16-rounded f32 vocabulary sum. Inductor currently materializes the expanded label-vocabulary one-hot tensor, reduces it back to a per-row scalar, applies the dense exp/update path, then schedules the two padded layout outputs and sibling vocabulary reduction as separate generic regions. It cannot do this today because scheduler/codegen does not canonicalize the guarded one-hot reduction into a label-indexed row scalar while also sinking the bf16 add into both materializations and the post-add column reduction. The fix is ALGEBRAIC_ELIMINATION plus SCHEDULER_FUSION: recognize this masked-LM backward template, avoid the dense one-hot producer, and generate one full-scope materialize-and-reduce plan with the exact cast boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 16384
SEQ = 512
VOCAB = 50265
VOCAB_PAD = 50272
PAD = 7


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
def _materialize_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    out_rows_ptr,
    out_trans_ptr,
    ROWS_N: tl.constexpr,
    SEQ_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    VOCAB_PAD_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS_N
    col_mask = cols < VOCAB_N
    mask = row_mask[:, None] & col_mask[None, :]

    label_offsets = rows + (rows // SEQ_N) + 1
    label = tl.load(labels_ptr + label_offsets, mask=row_mask, other=-100).to(tl.int64)
    active_label = label != -100
    safe_label = tl.where(active_label, label, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)

    numerator = tl.load(numerator_ptr).to(tl.float32)
    denominator = tl.load(denominator_ptr).to(tl.float32)
    scale = _f32_div(numerator, denominator)

    zero_f32 = tl.full((), 0.0, tl.float32)
    neg_one_f32 = tl.full((), -1.0, tl.float32)
    row_scale = tl.where(active_label, scale, zero_f32)

    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == zero_f32
    finite_row_sum = tl.where(in_vocab, _f32_mul(neg_one_f32, row_scale), zero_f32)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active_label, active_row_sum, zero_f32)

    one_hot = tl.where(safe_label[:, None] == cols[None, :], neg_one_f32, zero_f32)
    sparse_grad = _f32_mul(one_hot, row_scale[:, None])

    logits_offsets = rows[:, None] * VOCAB_PAD_N + cols[None, :]
    dense_offsets = rows[:, None] * VOCAB_N + cols[None, :]
    logits = tl.load(logits_ptr + logits_offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    centered = _f32_sub(logits, row_shift0[:, None])
    centered = _f32_sub(centered, row_shift1[:, None])
    exp_values = libdevice.exp(centered)

    correction = _f32_sub(sparse_grad, _f32_mul(exp_values, row_sum[:, None]))
    correction_bf16 = correction.to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.load(residual_ptr + dense_offsets, mask=mask, other=0.0)
    out_value = _bf16_add(residual, correction_bf16)

    tl.store(
        out_rows_ptr + rows[:, None] * VOCAB_PAD_N + cols[None, :],
        out_value,
        mask=mask,
    )
    tl.store(
        out_trans_ptr + cols[:, None] * ROWS_N + rows[None, :],
        tl.trans(out_value),
        mask=col_mask[:, None] & row_mask[None, :],
    )


@triton.jit
def _zero_pad_kernel(
    out_rows_ptr,
    out_trans_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    VOCAB_PAD_N: tl.constexpr,
    PAD_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < (ROWS_N * PAD_N)
    row = offsets // PAD_N
    pad_col = VOCAB_N + (offsets - row * PAD_N)
    zero = tl.zeros((BLOCK,), dtype=tl.float32).to(tl.bfloat16)

    tl.store(out_rows_ptr + row * VOCAB_PAD_N + pad_col, zero, mask=active)
    tl.store(out_trans_ptr + pad_col * ROWS_N + row, zero, mask=active)


@triton.jit
def _column_sum_kernel(
    out_rows_ptr,
    out_sum_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    VOCAB_PAD_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    col_block = tl.program_id(0)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    row_lanes = tl.arange(0, BLOCK_R)[None, :]
    col_mask = cols < VOCAB_N

    acc = tl.full((BLOCK_N, BLOCK_R), 0.0, tl.float32)
    for row_offset in tl.range(0, ROWS_N, BLOCK_R):
        rows = row_offset + row_lanes
        row_mask = rows < ROWS_N
        values = tl.load(
            out_rows_ptr + rows * VOCAB_PAD_N + cols,
            mask=col_mask & row_mask,
            other=0.0,
        ).to(tl.float32)
        acc = tl.where(col_mask & row_mask, _f32_add(acc, values), acc)

    sums = tl.sum(acc, axis=1)[:, None]
    rounded = sums.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_sum_ptr + cols, rounded, mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="020b9e29",
    BLOCK_M=16,
    BLOCK_N=256,
    FINAL_BLOCK_N=128,
    FINAL_BLOCK_R=64,
    materialize_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
    FINAL_BLOCK_R: int,
    materialize_warps: int,
    final_warps: int,
):
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
        _shape6,
        _shape7,
    ) = inputs

    out_rows = torch.empty_strided(
        (ROWS, VOCAB_PAD),
        (VOCAB_PAD, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    out_trans = torch.empty_strided(
        (VOCAB_PAD, ROWS),
        (ROWS, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    out_sum = torch.empty_strided((VOCAB,), (1,), device=logits.device, dtype=torch.float32)

    _materialize_kernel[(triton.cdiv(ROWS, BLOCK_M), triton.cdiv(VOCAB, BLOCK_N))](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_rows,
        out_trans,
        ROWS_N=ROWS,
        SEQ_N=SEQ,
        VOCAB_N=VOCAB,
        VOCAB_PAD_N=VOCAB_PAD,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=materialize_warps,
        num_stages=4,
    )
    _zero_pad_kernel[(triton.cdiv(ROWS * PAD, 1024),)](
        out_rows,
        out_trans,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        VOCAB_PAD_N=VOCAB_PAD,
        PAD_N=PAD,
        BLOCK=1024,
        num_warps=4,
    )
    _column_sum_kernel[(triton.cdiv(VOCAB, FINAL_BLOCK_N),)](
        out_rows,
        out_sum,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        VOCAB_PAD_N=VOCAB_PAD,
        BLOCK_N=FINAL_BLOCK_N,
        BLOCK_R=FINAL_BLOCK_R,
        num_warps=final_warps,
        num_stages=1,
    )
    return out_rows, out_trans, out_sum
