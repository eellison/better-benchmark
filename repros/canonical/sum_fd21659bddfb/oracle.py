"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GPT-Neo bf16 cross-entropy-backward dual-pad tail, including shifted labels from `arg2[:, 1:]`, scalar f32 division, ignore-index behavior, strided bf16 logits promotion, f32 `exp(logit - row_shift0 - row_shift1)`, explicit bf16 correction rounding, bf16 residual add, returned scalar zero, returned transposed padded `[50264, 4096]` bf16 output, and returned row-major padded `[4096, 50264]` bf16 output, while replacing the materialized one-hot tensor and its row sum with the equivalent guarded label scalar. Inductor currently scans each 50257-wide row to sum a one-hot tensor before rereading the same row for the exponential epilogue and separate pad/layout stores; it cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions into per-row guarded label formulas. The fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit a full-scope multi-output layout kernel for this backward pattern."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
SEQ_IN = 129
SEQ_OUT = 128
ROWS = BATCH * SEQ_OUT
VOCAB = 50257
PADDED_VOCAB = 50264
LOGITS_ROW_STRIDE = 50264
RESIDUAL_ROW_STRIDE = 50257


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
def _dual_pad_backward_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    zero_out_ptr,
    out_t_ptr,
    out_row_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    PADDED_N: tl.constexpr,
    SEQ_IN_N: tl.constexpr,
    SEQ_OUT_N: tl.constexpr,
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

    tl.store(zero_out_ptr, 0.0, mask=(col_block == 0) & (row_block == 0))

    scale_value = (
        tl.load(numerator_ptr).to(tl.float32)
        / tl.load(denominator_ptr).to(tl.float32)
    )
    label_offsets = 1 + (rows % SEQ_OUT_N) + SEQ_IN_N * (rows // SEQ_OUT_N)
    labels = tl.load(labels_ptr + label_offsets, mask=row_mask, other=-100).to(tl.int64)
    active = labels != -100
    safe_labels = tl.where(active, labels, 0)
    in_range = (safe_labels >= 0) & (safe_labels < VOCAB_N)

    scale_delta = _f32_sub(scale_value, scale_value)
    scale_is_finite = scale_delta == 0.0
    finite_row_sum = tl.where(in_range, -scale_value, 0.0)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, 0.0)
    row_scale = tl.where(active, scale_value, 0.0)

    one_hot = tl.where(safe_labels[:, None] == cols[None, :], -1.0, 0.0)
    one_hot_scaled = _f32_mul(one_hot, row_scale[:, None])

    logits_offsets = rows[:, None] * LOGITS_ROW_STRIDE_N + cols[None, :]
    residual_offsets = rows[:, None] * RESIDUAL_ROW_STRIDE_N + cols[None, :]
    logits = tl.load(logits_ptr + logits_offsets, mask=data_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=data_mask, other=0.0)
    row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(logits, row_shift0[:, None])
    centered = _f32_sub(centered, row_shift1[:, None])
    exp_values = libdevice.exp(centered)
    exp_times_sum = _f32_mul(exp_values, row_sum[:, None])
    correction = _f32_sub(one_hot_scaled, exp_times_sum).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    value = _bf16_add(residual, correction)
    value = tl.where(data_mask, value, 0.0)

    tl.store(out_row_ptr + rows[:, None] * PADDED_N + cols[None, :], value, mask=out_mask)
    tl.store(out_t_ptr + cols[None, :] * ROWS_N + rows[:, None], value, mask=out_mask)


# hf_GPTNeoForCausalLM train, rows=4096 vocab=50257 padded_vocab=50264.
@oracle_impl(
    hardware="B200",
    point="11dbdf2e",
    BLOCK_M=32,
    BLOCK_N=64,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        *_,
    ) = inputs

    zero_out = torch.empty_strided((), (), device=logits.device, dtype=torch.float32)
    out_t = torch.empty_strided(
        (PADDED_VOCAB, ROWS),
        (ROWS, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    out_row = torch.empty_strided(
        (ROWS, PADDED_VOCAB),
        (PADDED_VOCAB, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )

    _dual_pad_backward_kernel[
        (triton.cdiv(PADDED_VOCAB, BLOCK_N), triton.cdiv(ROWS, BLOCK_M))
    ](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        zero_out,
        out_t,
        out_row,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        PADDED_N=PADDED_VOCAB,
        SEQ_IN_N=SEQ_IN,
        SEQ_OUT_N=SEQ_OUT,
        LOGITS_ROW_STRIDE_N=LOGITS_ROW_STRIDE,
        RESIDUAL_ROW_STRIDE_N=RESIDUAL_ROW_STRIDE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return zero_out, out_t, out_row
