"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Blenderbot causal-LM cross-entropy-backward dense-gradient scope, including ignore-index label masking, scalar f32 division, bf16 rounding of the one-hot target term, bf16 rounding of shifted logits before libdevice natural exponential, bf16 gradient rounding before the incoming-gradient add, final bf16 output, and the returned transposed alias layout, while replacing the materialized one-hot row reduction with the equivalent guarded per-row label scalar. Inductor currently builds and scans a dense `[4096,8008]` one-hot tensor to recover a per-row scalar before rereading the same row for the exponential epilogue and bf16 store; it cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions through bf16 cast boundaries into per-row guarded label formulas. The fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit the dense bf16 epilogue directly with alias-return handling."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
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
def _one_hot_exp_add_kernel(
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
    row_active = rows < ROWS_N
    mask = row_active & (cols < VOCAB_N)

    raw_label = tl.load(labels_ptr + rows, mask=row_active, other=-100).to(tl.int64)
    valid_label = raw_label != -100
    safe_label = tl.where(valid_label, raw_label, 0)
    target_in_vocab = valid_label & (safe_label >= 0) & (safe_label < VOCAB_N)

    scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(tl.float32)
    zero = tl.full((BLOCK_M, 1), 0.0, tl.float32)
    neg_one = tl.full((BLOCK_M, 1), -1.0, tl.float32)
    row_scale = tl.where(valid_label, scale, zero)

    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == 0.0
    rounded_target_sum = _f32_mul(neg_one, row_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    finite_row_sum = tl.where(target_in_vocab, rounded_target_sum, zero)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(valid_label, active_row_sum, zero)

    one_hot = tl.where(safe_label == cols, neg_one, zero)
    one_hot_scaled = _f32_mul(one_hot, row_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)

    offsets = rows * VOCAB_N + cols
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    shift0 = tl.load(row_shift0_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    shift1 = tl.load(row_shift1_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
    incoming = tl.load(incoming_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    shifted = _f32_sub(_f32_sub(logits, shift0), shift1).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    exp_values = libdevice.exp(shifted)
    exp_times_sum = _f32_mul(exp_values, row_sum)
    delta = _f32_sub(one_hot_scaled, exp_times_sum).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    out = _f32_add(incoming, delta).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + offsets, out, mask=mask)


# bb71d599: BlenderbotForCausalLM train, rows=4096 vocab=8008 bf16 dense gradient.
@oracle_impl(hardware="B200", point="bb71d599", BLOCK_M=1, BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    numerator, denominator, labels, logits, row_shift0, row_shift1, incoming_grad, *_ = inputs

    out = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    _one_hot_exp_add_kernel[(triton.cdiv(ROWS, BLOCK_M), triton.cdiv(VOCAB, BLOCK_N))](
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
