"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GPT-J bf16 layer-norm-backward/scatter scope, including the four bf16 addends, both returned `[4096]` reductions, row-local summaries, the residual update, and the dense `[50400,4096]` `_unsafe_masked_index_put_accumulate` output with the exact valid-index mask, whereas Inductor schedules a separate column-reduction kernel, dense zero-fill, and indexed scatter/reduction kernel around the same mandatory dense output; Inductor cannot materially beat this full scope today because the returned scatter tensor is a real contiguous 826 MB f32 materialization whose zero-fill dominates runtime, so the remaining fusion opportunity is hidden by output bandwidth; the fix is BANDWIDTH_BOUND: keep this as an at-floor proof unless broader dense-output byte accounting or scatter materialization elimination becomes available."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 4096
VOCAB = 50400
BLOCK_H = 1024
H_CHUNKS = HIDDEN // BLOCK_H
BLOCK_ZERO = 2048


@triton.jit
def _producer_sum(a_ptr, b_ptr, c_ptr, d_ptr, offsets, active):
    a = tl.load(a_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    d = tl.load(d_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    return ((a + b) + c) + d


@triton.jit
def _zero_and_row_partials_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    weight_ptr,
    src_ptr,
    mean_ptr,
    scale_ptr,
    scatter_out_ptr,
    sum_prod_out_ptr,
    sum_x_out_ptr,
    row_sum_partials_ptr,
    row_dot_partials_ptr,
    TOTAL_SCATTER: tl.constexpr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_H_: tl.constexpr,
    H_CHUNKS_: tl.constexpr,
    BLOCK_ZERO_: tl.constexpr,
):
    pid = tl.program_id(0)

    zero_offsets = pid * BLOCK_ZERO_ + tl.arange(0, BLOCK_ZERO_)
    tl.store(
        scatter_out_ptr + zero_offsets,
        tl.zeros((BLOCK_ZERO_,), tl.float32),
        mask=zero_offsets < TOTAL_SCATTER,
    )

    partial_programs: tl.constexpr = ROWS_ * H_CHUNKS_
    if pid < partial_programs:
        row = pid // H_CHUNKS_
        chunk = pid - row * H_CHUNKS_
        cols = chunk * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        col_mask = cols < HIDDEN_
        offsets = row * HIDDEN_ + cols

        summed = _producer_sum(a_ptr, b_ptr, c_ptr, d_ptr, offsets, col_mask)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        src = tl.load(src_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)
        centered_scaled = (src - mean) * scale
        weighted = summed * weight

        partial_offset = row * H_CHUNKS_ + chunk
        tl.store(row_sum_partials_ptr + partial_offset, tl.sum(weighted, axis=0))
        tl.store(
            row_dot_partials_ptr + partial_offset,
            tl.sum(weighted * centered_scaled, axis=0),
        )

        if row == 0:
            zeros = tl.zeros((BLOCK_H_,), tl.float32)
            tl.store(sum_prod_out_ptr + cols, zeros, mask=col_mask)
            tl.store(sum_x_out_ptr + cols, zeros, mask=col_mask)


@triton.jit
def _scatter_and_column_reduce_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    weight_ptr,
    src_ptr,
    mean_ptr,
    scale_ptr,
    residual_ptr,
    index_ptr,
    sum_prod_out_ptr,
    sum_x_out_ptr,
    scatter_out_ptr,
    row_sum_partials_ptr,
    row_dot_partials_ptr,
    HIDDEN_: tl.constexpr,
    VOCAB_: tl.constexpr,
    BLOCK_H_: tl.constexpr,
    H_CHUNKS_: tl.constexpr,
    INV_HIDDEN_: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK_H_ + tl.arange(0, BLOCK_H_)
    col_mask = cols < HIDDEN_
    offsets = row * HIDDEN_ + cols

    summed = _producer_sum(a_ptr, b_ptr, c_ptr, d_ptr, offsets, col_mask)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    src = tl.load(src_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row).to(tl.float32)
    scale = tl.load(scale_ptr + row).to(tl.float32)
    centered_scaled = (src - mean) * scale
    weighted = summed * weight

    chunks = tl.arange(0, H_CHUNKS_)
    partial_offsets = row * H_CHUNKS_ + chunks
    row_sum = tl.sum(tl.load(row_sum_partials_ptr + partial_offsets), axis=0)
    row_dot = tl.sum(tl.load(row_dot_partials_ptr + partial_offsets), axis=0)

    tl.atomic_add(sum_prod_out_ptr + cols, summed * centered_scaled, sem="relaxed", mask=col_mask)
    tl.atomic_add(sum_x_out_ptr + cols, summed, sem="relaxed", mask=col_mask)

    correction = (weighted * HIDDEN_ - row_sum - centered_scaled * row_dot) * (scale * INV_HIDDEN_)
    update = tl.load(residual_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32) + correction

    raw_index = tl.load(index_ptr + row).to(tl.int64)
    valid_index = (raw_index >= 0) & (raw_index < VOCAB_) & (raw_index != -1)
    scatter_offsets = raw_index * HIDDEN_ + cols
    tl.atomic_add(
        scatter_out_ptr + scatter_offsets,
        update,
        sem="relaxed",
        mask=col_mask & valid_index,
    )


@oracle_impl(
    hardware="B200",
    point="cce081e2",
    block_h=1024,
    block_zero=2048,
    zero_warps=8,
    scatter_warps=8,
)
def oracle_forward(
    inputs,
    *,
    block_h: int,
    block_zero: int,
    zero_warps: int,
    scatter_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        src,
        mean,
        scale,
        residual,
        index,
        *_shape_params,
    ) = inputs

    device = arg0.device
    h_chunks = triton.cdiv(HIDDEN, block_h)
    total_scatter = VOCAB * HIDDEN
    scatter_out = torch.empty_strided(
        (VOCAB, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_prod_out = torch.empty_strided((HIDDEN,), (1,), device=device, dtype=torch.float32)
    sum_x_out = torch.empty_strided((HIDDEN,), (1,), device=device, dtype=torch.float32)
    row_sum_partials = torch.empty_strided(
        (ROWS, h_chunks),
        (h_chunks, 1),
        device=device,
        dtype=torch.float32,
    )
    row_dot_partials = torch.empty_strided(
        (ROWS, h_chunks),
        (h_chunks, 1),
        device=device,
        dtype=torch.float32,
    )

    zero_blocks = triton.cdiv(total_scatter, block_zero)
    partial_blocks = ROWS * h_chunks
    _zero_and_row_partials_kernel[(max(zero_blocks, partial_blocks),)](
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        src,
        mean,
        scale,
        scatter_out,
        sum_prod_out,
        sum_x_out,
        row_sum_partials,
        row_dot_partials,
        TOTAL_SCATTER=total_scatter,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_H_=block_h,
        H_CHUNKS_=h_chunks,
        BLOCK_ZERO_=block_zero,
        num_warps=zero_warps,
        num_stages=3,
    )
    _scatter_and_column_reduce_kernel[(ROWS, h_chunks)](
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        src,
        mean,
        scale,
        residual,
        index,
        sum_prod_out,
        sum_x_out,
        scatter_out,
        row_sum_partials,
        row_dot_partials,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        BLOCK_H_=block_h,
        H_CHUNKS_=h_chunks,
        INV_HIDDEN_=1.0 / HIDDEN,
        num_warps=scatter_warps,
        num_stages=3,
    )
    return sum_prod_out, sum_x_out, scatter_out
