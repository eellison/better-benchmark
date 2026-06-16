"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Electra bf16 shifted causal-LM cross-entropy-backward fragment, including ignore-index label handling, scalar f32 division, natural `exp(logit - row_shift0 - row_shift1)` from the padded-stride logits input, bf16 correction/add rounding, the returned right-padded dense output, the separate returned transposed unpadded view, and the bf16-rounded f32 vocabulary sum. Inductor currently materializes the expanded one-hot path and schedules the dense add, padding/layout output, transposed view, and sibling vocabulary reduction as separate generic regions; it cannot do better on this isolated scope without recognizing the sparse-label row scalar and sinking the bf16 add into both materialization and reduction consumers. The fix is BANDWIDTH_BOUND plus SCHEDULER_FUSION: recognize masked-LM backward as a sparse-label dense materialize-and-reduce template that avoids the one-hot scan while still preserving the required dense reads, libdevice exp, output layout/aliasing, and exact bf16 reduction boundary."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 64
SEQ_IN = 513
SEQ_OUT = 512
ROWS = BATCH * SEQ_OUT
VOCAB = 30522
VOCAB_PAD = 30528
PAD = VOCAB_PAD - VOCAB


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _materialize_and_partial_sum_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    out_padded_ptr,
    out_base_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    VOCAB_PAD_N: tl.constexpr,
    SEQ_IN_N: tl.constexpr,
    SEQ_OUT_N: tl.constexpr,
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

    scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(
        tl.float32
    )
    label_offsets = (rows // SEQ_OUT_N) * SEQ_IN_N + (rows - (rows // SEQ_OUT_N) * SEQ_OUT_N) + 1
    raw_label = tl.load(labels_ptr + label_offsets, mask=row_mask, other=-100).to(
        tl.int64
    )
    active_label = raw_label != -100
    safe_label = tl.where(active_label, raw_label, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)

    row_scale = tl.where(active_label, scale, 0.0)
    scale_delta = _f32_sub(scale, scale)
    scale_is_finite = scale_delta == 0.0
    finite_row_sum = tl.where(active_label & in_vocab, -row_scale, 0.0)
    row_sum = tl.where(
        active_label,
        tl.where(scale_is_finite, finite_row_sum, scale_delta),
        0.0,
    )

    one_hot = tl.where(safe_label[:, None] == cols[None, :], -1.0, 0.0)
    one_hot_scaled = _f32_mul(one_hot, row_scale[:, None])

    logits_offsets = rows[:, None] * VOCAB_PAD_N + cols[None, :]
    dense_offsets = rows[:, None] * VOCAB_N + cols[None, :]
    logits = tl.load(logits_ptr + logits_offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(
        tl.float32
    )
    row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(
        tl.float32
    )

    centered = _f32_sub(logits, row_shift0[:, None])
    centered = _f32_sub(centered, row_shift1[:, None])
    exp_values = libdevice.exp(centered)
    correction = _f32_sub(one_hot_scaled, _f32_mul(exp_values, row_sum[:, None]))
    correction_bf16 = correction.to(tl.bfloat16, fp_downcast_rounding="rtne")

    residual = tl.load(residual_ptr + dense_offsets, mask=mask, other=0.0)
    out_value = _bf16_add(residual, correction_bf16)
    tl.store(out_base_ptr + dense_offsets, out_value, mask=mask)
    tl.store(out_padded_ptr + logits_offsets, out_value, mask=mask)


@triton.jit
def _zero_pad_kernel(
    out_padded_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    VOCAB_PAD_N: tl.constexpr,
    PAD_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < ROWS_N * PAD_N
    row = offsets // PAD_N
    pad_col = VOCAB_N + (offsets - row * PAD_N)
    zero = tl.zeros((BLOCK,), dtype=tl.float32).to(tl.bfloat16)
    tl.store(out_padded_ptr + row * VOCAB_PAD_N + pad_col, zero, mask=active)


@triton.jit
def _column_sum_kernel(
    out_base_ptr,
    out_sum_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    row_lanes = tl.arange(0, BLOCK_R)[None, :]
    col_mask = cols < VOCAB_N

    acc = tl.full((BLOCK_C, BLOCK_R), 0.0, tl.float32)
    for row_offset in tl.range(0, ROWS_N, BLOCK_R):
        rows = row_offset + row_lanes
        row_mask = rows < ROWS_N
        values = tl.load(
            out_base_ptr + rows * VOCAB_N + cols,
            mask=col_mask & row_mask,
            other=0.0,
        ).to(tl.float32)
        acc = tl.where(col_mask & row_mask, _f32_add(acc, values), acc)

    sums = tl.sum(acc, axis=1)
    rounded = sums.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_sum_ptr + cols, rounded[:, None], mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="d82d2bc6",
    BLOCK_M=8,
    BLOCK_N=512,
    FINAL_BLOCK_C=128,
    FINAL_BLOCK_R=64,
    materialize_warps=4,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_C: int,
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
    ) = inputs

    device = logits.device
    out_padded = torch.empty_strided(
        (ROWS, VOCAB_PAD),
        (VOCAB_PAD, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_base = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_sum = torch.empty_strided((VOCAB,), (1,), device=device, dtype=torch.float32)

    num_row_tiles = triton.cdiv(ROWS, BLOCK_M)

    _materialize_and_partial_sum_kernel[
        (num_row_tiles, triton.cdiv(VOCAB, BLOCK_N))
    ](
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        out_padded,
        out_base,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        VOCAB_PAD_N=VOCAB_PAD,
        SEQ_IN_N=SEQ_IN,
        SEQ_OUT_N=SEQ_OUT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=materialize_warps,
        num_stages=4,
    )
    _zero_pad_kernel[(triton.cdiv(ROWS * PAD, 1024),)](
        out_padded,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        VOCAB_PAD_N=VOCAB_PAD,
        PAD_N=PAD,
        BLOCK=1024,
        num_warps=4,
    )
    _column_sum_kernel[(triton.cdiv(VOCAB, FINAL_BLOCK_C),)](
        out_base,
        out_sum,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_R=FINAL_BLOCK_R,
        num_warps=final_warps,
        num_stages=1,
    )
    return out_padded, out_base.permute(1, 0), out_sum
