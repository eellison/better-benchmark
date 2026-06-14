"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete M2M100 bf16 cross-entropy-backward materialization scope in one Triton kernel, including ignore-index label handling, scalar f32 division, the one-hot label-gradient producer with bf16 rounding before its row sum, natural `exp` of the bf16-rounded shifted logits, bf16 delta rounding, bf16 residual add, the contiguous `[8192,128112]` output, and the returned `[128112,8192]` transpose alias, while replacing the dense one-hot row reduction with the equivalent guarded per-row scalar. Inductor currently lowers the decomposed `iota == label` expansion and one-hot sum as a generic full-vocabulary reduction before rereading the same rows for the exponential epilogue and output store; it cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions through bf16 cast boundaries into per-row guarded label formulas. The fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit the dense bf16 epilogue directly with alias-return handling."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 8192
VOCAB = 128112
OUT_SHAPE = (ROWS, VOCAB)
OUT_STRIDE = (VOCAB, 1)
PERMUTE_SHAPE = (VOCAB, ROWS)
PERMUTE_STRIDE = (1, VOCAB)


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
def _m2m100_loss_backward_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    shift0_ptr,
    shift1_ptr,
    residual_ptr,
    out_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS_N
    col_mask = cols < VOCAB_N
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * VOCAB_N + cols[None, :]

    zero = tl.full((), 0.0, tl.float32)
    scale = _f32_div(
        tl.load(numerator_ptr).to(tl.float32),
        tl.load(denominator_ptr).to(tl.float32),
    )
    label = tl.load(labels_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
    active = label != -100
    safe_label = tl.where(active, label, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)
    row_scale = tl.where(active, scale, zero)

    one_hot = tl.where(safe_label[:, None] == cols[None, :], -1.0, 0.0)
    label_grad = _f32_mul(one_hot, row_scale[:, None]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)

    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == zero
    rounded_neg_scale = _f32_mul(-1.0, scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    finite_row_sum = tl.where(in_vocab, rounded_neg_scale, zero)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero)

    logits = tl.load(
        logits_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    shift0 = tl.load(shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    shift1 = tl.load(shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    shifted = _f32_sub(_f32_sub(logits, shift0[:, None]), shift1[:, None])
    shifted = shifted.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    exp_shifted = libdevice.exp(shifted)

    exp_times_sum = _f32_mul(exp_shifted, row_sum[:, None])
    delta = _f32_sub(label_grad, exp_times_sum).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    out = _f32_add(residual, delta.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="46e5658a", BLOCK_M=8, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        numerator,
        denominator,
        labels,
        logits,
        shift0,
        shift1,
        residual,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs
    del shape0, shape1, shape2, shape3, shape4

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=logits.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(ROWS, BLOCK_M), triton.cdiv(VOCAB, BLOCK_N))
    _m2m100_loss_backward_kernel[grid](
        numerator,
        denominator,
        labels,
        logits,
        shift0,
        shift1,
        residual,
        out,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, out.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE)
