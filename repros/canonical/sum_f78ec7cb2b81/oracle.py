"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Blenderbot cross-entropy-backward dense-gradient scope, including label masking, scalar f32 division, libdevice natural exponential, f32 residual add, final bf16 cast, returned contiguous `[2048, 8008]` output, and returned transposed alias layout, while replacing the materialized one-hot row reduction with the equivalent guarded label scalar. Inductor currently scans each 8008-wide row to sum a one-hot tensor before rereading the same row for the exponential epilogue and bf16 store; it cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions into per-row guarded label formulas. The fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit the dense bf16 epilogue directly with alias-return handling."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
ROWS = BATCH * SEQ
VOCAB = 8008


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
def _one_hot_exp_bf16_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    incoming_grad_ptr,
    out_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < ROWS_N
    mask = row_mask & (cols < VOCAB_N)

    raw_label = tl.load(labels_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
    active = raw_label != -100
    selected_label = tl.where(active, raw_label, 0)

    scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(tl.float32)
    zero = tl.full((BLOCK_M, 1), 0.0, tl.float32)
    neg_one = tl.full((BLOCK_M, 1), -1.0, tl.float32)
    row_scale = tl.where(active, scale, zero)

    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == 0.0
    target_in_vocab = active & (selected_label >= 0) & (selected_label < VOCAB_N)
    finite_row_sum = tl.where(target_in_vocab, _f32_mul(neg_one, row_scale), zero)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero)

    one_hot = tl.where(selected_label == cols, neg_one, zero)
    one_hot_scaled = _f32_mul(one_hot, row_scale)

    offsets = rows * VOCAB_N + cols
    incoming = tl.load(incoming_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    shifted = _f32_sub(_f32_sub(logits, shift0), shift1)
    exp_values = libdevice.exp(shifted)
    exp_times_sum = _f32_mul(exp_values, row_sum)
    delta = _f32_sub(one_hot_scaled, exp_times_sum)
    out = _f32_add(incoming, delta).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + offsets, out, mask=mask)


# BlenderbotForConditionalGeneration train/infer, rows=2048 vocab=8008.
@oracle_impl(hardware="B200", point="3cca3d26", BLOCK_M=1, BLOCK_N=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    numerator, denominator, labels, logits, row_shift0, row_shift1, incoming_grad, *_ = inputs

    out = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    _one_hot_exp_bf16_kernel[(triton.cdiv(ROWS, BLOCK_M), triton.cdiv(VOCAB, BLOCK_N))](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        incoming_grad,
        out,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out, torch.as_strided(out, (VOCAB, ROWS), (1, VOCAB))
