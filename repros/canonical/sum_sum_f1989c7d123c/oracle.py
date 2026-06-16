"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GoogleFnet masked-LM cross-entropy-backward fragment, including ignore-index label handling, scalar f32 division, natural `exp(logit - row_shift0 - row_shift1)`, residual add, the returned dense `[16384,32000]` buffer, its transposed alias, and the returned `[32000]` column sum, while replacing the materialized one-hot tensor and its row sum with the equivalent guarded label scalar before running the same dense materialization and column-reduction shape; Inductor currently scans each 32000-wide one-hot row to rediscover a scalar known from the label and then separately reduces the full dense buffer for the sibling column sum, but the required dense input reads, libdevice exp work, output materialization, and exact column reduction dominate the floor; the fix is BANDWIDTH_BOUND: record this repro as an at-floor dense backward/materialize-plus-reduce instance unless broader memory-traffic or reduction-throughput improvements move both paths."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
VOCAB = 32000
ROW_BLOCK = 16


@triton.jit
def _materialize_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    row_shift0_ptr,
    row_shift1_ptr,
    residual_ptr,
    out_base_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
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

    raw_label = tl.load(labels_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
    active = raw_label != -100
    safe_label = tl.where(active, raw_label, 0)

    numerator = tl.load(numerator_ptr).to(tl.float32)
    denominator = tl.load(denominator_ptr).to(tl.float32)
    scale = numerator / denominator

    zero_f32 = tl.full((), 0.0, tl.float32)
    neg_one_f32 = tl.full((), -1.0, tl.float32)
    row_scale = tl.where(active, scale, zero_f32)

    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)
    scale_delta = scale - scale
    scale_is_finite = scale_delta == zero_f32
    finite_row_sum = tl.where(in_vocab, neg_one_f32 * row_scale, zero_f32)
    active_row_sum = tl.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = tl.where(active, active_row_sum, zero_f32)

    one_hot = tl.where(safe_label[:, None] == cols[None, :], neg_one_f32, zero_f32)
    one_hot_scaled = one_hot * row_scale[:, None]

    offsets = rows[:, None] * VOCAB_N + cols[None, :]
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift0 = tl.load(row_shift0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_shift1 = tl.load(row_shift1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    centered = logits - row_shift0[:, None]
    centered = centered - row_shift1[:, None]
    exp_values = libdevice.exp(centered)
    correction = one_hot_scaled - exp_values * row_sum[:, None]
    values = residual + correction
    values = tl.where(mask, values, zero_f32)

    tl.store(out_base_ptr + offsets, values, mask=mask)


@triton.jit
def _column_sum_from_base_kernel(
    out_base_ptr,
    out_sum_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
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
            out_base_ptr + rows * VOCAB_N + cols,
            mask=col_mask & row_mask,
            other=0.0,
        ).to(tl.float32)
        next_acc = acc + values
        acc = tl.where(col_mask & row_mask, next_acc, acc)

    sums = tl.sum(acc, axis=1)[:, None]
    tl.store(out_sum_ptr + cols, sums, mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="8c80d3e5",
    BLOCK_M=16,
    BLOCK_N=256,
    FINAL_BLOCK_N=64,
    FINAL_BLOCK_R=64,
    materialize_warps=8,
    final_warps=16,
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
    ) = inputs

    out_base = torch.empty_strided(
        (ROWS, VOCAB),
        (VOCAB, 1),
        device=logits.device,
        dtype=torch.float32,
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
        out_base,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=materialize_warps,
        num_stages=4,
    )

    _column_sum_from_base_kernel[(triton.cdiv(VOCAB, FINAL_BLOCK_N),)](
        out_base,
        out_sum,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_N=FINAL_BLOCK_N,
        BLOCK_R=FINAL_BLOCK_R,
        num_warps=final_warps,
        num_stages=1,
    )

    return out_base, out_base.permute(1, 0), out_sum
