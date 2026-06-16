"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MegatronBERT bf16 causal-LM cross-entropy-backward fragment, including shifted-label ignore-index handling, scalar f32 division, natural `exp((logit - row_shift0) - row_shift1)`, bf16 correction rounding, bf16 residual add, the returned dense `[8192, 29056]` buffer, its metadata-only transposed view, and the returned bf16-rounded f32 vocabulary sum. Inductor currently materializes the expanded equality-built one-hot tensor, reduces it to rediscover a per-row scalar, applies the dense exp/update path, and separately reduces the final dense buffer. It cannot do this today because algebraic simplification does not canonicalize the one-hot row reduction into a guarded label scalar and keep the resulting producer shared with the output and column-reduction consumers. The fix is ALGEBRAIC_ELIMINATION plus SCHEDULER_FUSION: rewrite the one-hot path to a sparse label correction and emit a fused materialize-plus-column-partial plan with exact bf16 cast boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 16
SEQ_IN = 513
SEQ_OUT = 512
ROWS = BATCH * SEQ_OUT
VOCAB = 29056


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
def _materialize_partial_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    out_base_ptr,
    partial_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    SEQ_IN_N: tl.constexpr,
    SEQ_OUT_N: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS_N
    col_mask = cols < VOCAB_N
    mask = row_mask[:, None] & col_mask[None, :]

    label_offsets = (rows // SEQ_OUT_N) * SEQ_IN_N + (rows % SEQ_OUT_N) + 1
    label = tl.load(labels_ptr + label_offsets, mask=row_mask, other=-100).to(tl.int64)
    active = label != -100
    safe_label = tl.where(active, label, 0)

    scale = _f32_div(tl.load(numerator_ptr).to(tl.float32), tl.load(denominator_ptr).to(tl.float32))
    zero_f32 = tl.full((), 0.0, tl.float32)
    neg_one_f32 = tl.full((), -1.0, tl.float32)
    row_scale = tl.where(active, scale, zero_f32)

    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)
    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == zero_f32
    finite_row_sum = tl.where(in_vocab, _f32_mul(neg_one_f32, row_scale), zero_f32)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero_f32)

    one_hot = tl.where(safe_label[:, None] == cols[None, :], neg_one_f32, zero_f32)
    sparse_grad = _f32_mul(one_hot, row_scale[:, None])

    offsets = rows[:, None] * VOCAB_N + cols[None, :]
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    centered = _f32_sub(logits, row_shift0[:, None])
    centered = _f32_sub(centered, row_shift1[:, None])
    exp_values = libdevice.exp(centered)

    correction = _f32_sub(sparse_grad, _f32_mul(exp_values, row_sum[:, None]))
    correction_bf16 = correction.to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    out_value = _bf16_add(residual, correction_bf16)

    tl.store(out_base_ptr + offsets, out_value, mask=mask)
    partial = tl.sum(tl.where(mask, out_value.to(tl.float32), 0.0), axis=0)
    tl.store(
        partial_ptr + cols * NUM_ROW_TILES + row_tile,
        partial,
        mask=col_mask,
    )


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    out_sum_ptr,
    VOCAB_N: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (cols[:, None] < VOCAB_N) & (tiles[None, :] < NUM_ROW_TILES)
    values = tl.load(
        partial_ptr + cols[:, None] * NUM_ROW_TILES + tiles[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=1)
    rounded = sums.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_sum_ptr + cols, rounded, mask=cols < VOCAB_N)


@oracle_impl(
    hardware="B200",
    point="7b151c28",
    BLOCK_M=64,
    BLOCK_N=128,
    FINAL_BLOCK_C=8,
    materialize_warps=8,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_C: int,
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
    ) = inputs

    out_base = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    out_sum = torch.empty_strided((VOCAB,), (1,), device=logits.device, dtype=torch.float32)
    num_row_tiles = triton.cdiv(ROWS, BLOCK_M)
    partial = torch.empty_strided(
        (VOCAB, num_row_tiles),
        (num_row_tiles, 1),
        device=logits.device,
        dtype=torch.float32,
    )

    _materialize_partial_kernel[(num_row_tiles, triton.cdiv(VOCAB, BLOCK_N))](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_base,
        partial,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        SEQ_IN_N=SEQ_IN,
        SEQ_OUT_N=SEQ_OUT,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=materialize_warps,
        num_stages=4,
    )
    _finalize_sum_kernel[(triton.cdiv(VOCAB, FINAL_BLOCK_C),)](
        partial,
        out_sum,
        VOCAB_N=VOCAB,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        num_warps=final_warps,
        num_stages=1,
    )
    return out_base, out_base.permute(1, 0), out_sum
