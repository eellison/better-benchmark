"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DistillGPT2 bf16 cross-entropy-backward dense-gradient scope, including shifted label slice semantics, scalar f32 division, ignore-index masking, strided bf16 logits promotion, f32 `exp(logit - row_shift0 - row_shift1)`, final contiguous `[32,512,50257]` bf16 output, and the required output dtype/stride, while replacing the materialized one-hot row reduction with the equivalent guarded label scalar. Inductor currently expands the one-hot tensor and scans each 50257-wide row to recover a scalar known from the label before rereading the logits row for the exponential epilogue; it cannot do this today because algebraic simplification does not canonicalize masked one-hot reductions into per-row guarded formulas. The fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit the dense bf16 epilogue directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
SEQ_OUT = 512
SEQ_IN = 513
ROWS = BATCH * SEQ_OUT
VOCAB = 50257
LOGITS_ROW_STRIDE = 50264
OUT_SHAPE = (BATCH, SEQ_OUT, VOCAB)
OUT_STRIDE = (SEQ_OUT * VOCAB, VOCAB, 1)


@triton.jit
def _distillgpt2_ce_backward_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    out_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    LOGITS_ROW_STRIDE_N: tl.constexpr,
    SEQ_OUT_N: tl.constexpr,
    SEQ_IN_N: tl.constexpr,
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

    scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(
        tl.float32
    )

    label_offsets = 1 + (rows % SEQ_OUT_N) + SEQ_IN_N * (rows // SEQ_OUT_N)
    raw_labels = tl.load(labels_ptr + label_offsets, mask=row_mask, other=-100).to(tl.int64)
    active = raw_labels != -100
    safe_labels = tl.where(active, raw_labels, 0)
    in_vocab = (safe_labels >= 0) & (safe_labels < VOCAB_N)

    zero = tl.full((), 0.0, tl.float32)
    row_scale = tl.where(active, scale, zero)

    scale_delta = scale - scale
    scale_is_finite = scale_delta == zero
    finite_row_sum = tl.where(in_vocab, -scale, zero)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero)

    one_hot = tl.where(safe_labels[:, None] == cols[None, :], -1.0, 0.0)
    one_hot_scaled = one_hot * row_scale[:, None]

    logits_offsets = rows[:, None] * LOGITS_ROW_STRIDE_N + cols[None, :]
    out_offsets = rows[:, None] * VOCAB_N + cols[None, :]
    logits = tl.load(logits_ptr + logits_offsets, mask=mask, other=0.0).to(tl.float32)
    shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    shifted = (logits - shift0[:, None]) - shift1[:, None]
    exp_values = libdevice.exp(shifted)
    exp_times_sum = exp_values * row_sum[:, None]
    out = (one_hot_scaled - exp_times_sum).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + out_offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="ceaa9c1c", BLOCK_M=1, BLOCK_N=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    numerator, denominator, labels, logits, row_shift0, row_shift1, shape0, shape1, shape2, shape3 = inputs
    del shape0, shape1, shape2, shape3

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=logits.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(ROWS, BLOCK_M), triton.cdiv(VOCAB, BLOCK_N))
    _distillgpt2_ce_backward_kernel[grid](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        out,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        LOGITS_ROW_STRIDE_N=LOGITS_ROW_STRIDE,
        SEQ_OUT_N=SEQ_OUT,
        SEQ_IN_N=SEQ_IN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
