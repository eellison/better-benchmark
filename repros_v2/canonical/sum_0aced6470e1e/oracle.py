"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete OPT bf16 cross-entropy-backward materialization scope in one Triton kernel, including shifted-label slice/clone/view semantics from the padded-stride `arg2[:, 1:]`, ignore-index handling, scalar f32 division, f32 one-hot label-gradient production and row sum, natural `exp(logit - row_shift0 - row_shift1)`, bf16 correction rounding, bf16 residual add, the contiguous `[8192,50272]` output, and the returned `[50272,8192]` transpose alias, while replacing the dense one-hot row reduction with the equivalent guarded per-row scalar. Inductor currently lowers the decomposed shifted-label `iota == label` expansion and one-hot sum as a generic full-vocabulary reduction before rereading the same rows for the exponential epilogue and output store; it cannot do this today because algebraic simplification does not canonicalize one-hot masked reductions into per-row guarded label formulas while preserving padded shifted-label indexing and dtype boundaries. The fix is ALGEBRAIC_ELIMINATION: add a guarded one-hot reduction rewrite before reduction scheduling and emit the dense bf16 epilogue directly with alias-return handling."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 4
SEQ = 2048
ROWS = BATCH * SEQ
VOCAB = 50272
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
def _opt_loss_backward_row_loop_kernel(
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
    SEQ_N: tl.constexpr,
    LABEL_ROW_STRIDE_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols_base = tl.arange(0, BLOCK_N)

    label_offset = (row // SEQ_N) * LABEL_ROW_STRIDE_N + (row % SEQ_N) + 1
    label = tl.load(labels_ptr + label_offset).to(tl.int64)
    active = label != -100
    safe_label = tl.where(active, label, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)

    zero = tl.full((), 0.0, tl.float32)
    scale = _f32_div(
        tl.load(numerator_ptr).to(tl.float32),
        tl.load(denominator_ptr).to(tl.float32),
    )
    row_scale = tl.where(active, scale, zero)

    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == zero
    finite_row_sum = tl.where(in_vocab, _f32_mul(-1.0, scale), zero)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero)

    shift0 = tl.load(shift0_ptr + row).to(tl.float32)
    shift1 = tl.load(shift1_ptr + row).to(tl.float32)

    for col_start in tl.range(0, VOCAB_N, BLOCK_N):
        cols = col_start + cols_base
        mask = cols < VOCAB_N
        offsets = row * VOCAB_N + cols

        one_hot = tl.where(safe_label == cols, -1.0, 0.0)
        one_hot_scaled = _f32_mul(one_hot, row_scale)
        logits = tl.load(
            logits_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        shifted = _f32_sub(_f32_sub(logits, shift0), shift1)
        exp_shifted = libdevice.exp(shifted)
        exp_times_sum = _f32_mul(exp_shifted, row_sum)
        correction = _f32_sub(one_hot_scaled, exp_times_sum).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        out = _f32_add(residual, correction.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="1d8e61f8", BLOCK_N=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
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
    _opt_loss_backward_row_loop_kernel[(ROWS,)](
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
        SEQ_N=SEQ,
        LABEL_ROW_STRIDE_N=int(labels.stride(0)),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out, out.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE)
