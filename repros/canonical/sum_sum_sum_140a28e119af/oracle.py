"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete MegatronBERT bf16 embedding/layernorm-backward return tuple by reconstructing the shared three-input bf16-to-fp32 row producer once, accumulating both returned `[1024]` hidden reductions from cooperative row partials, and directly applying the position, segment, and vocabulary duplicate-preserving `_unsafe_masked_index_put_accumulate` outputs, whereas Inductor materializes the full `[16,512,1024]` producer and lowers the sibling reductions, zero fills, validity masks, dense vocabulary base add, and three indexed accumulators as separate generic regions; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that keeps row-local layernorm-backward reductions live while feeding multiple indexed destinations with different accumulator semantics and bf16/fp32 conversion boundaries; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that initializes dense destinations directly, folds the masks into indexed updates, shares the row producer across scatter outputs, and finalizes compatible hidden-column reductions from common partials."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 1024
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 29056
DROPOUT_SCALE = 1.1111111111111112
ROW_FACTOR = 1024.0


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _init_outputs_kernel(
    base_vocab_ptr,
    position_out_ptr,
    segment_out_ptr,
    vocab_out_ptr,
    TOTAL_POSITION: tl.constexpr,
    TOTAL_SEGMENT: tl.constexpr,
    TOTAL_VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)

    vocab_mask = offsets < TOTAL_VOCAB
    base = tl.load(base_vocab_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(vocab_out_ptr + offsets, base, mask=vocab_mask)
    tl.store(position_out_ptr + offsets, zeros, mask=offsets < TOTAL_POSITION)
    tl.store(segment_out_ptr + offsets, zeros, mask=offsets < TOTAL_SEGMENT)


@triton.jit
def _row_partials_scatter_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    weight_ptr,
    norm_ptr,
    scale_ptr,
    residual_ptr,
    keep_ptr,
    position_idx_ptr,
    segment_idx_ptr,
    vocab_idx_ptr,
    position_out_ptr,
    segment_out_ptr,
    vocab_out_ptr,
    partials_ptr,
    producer_tmp_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32)
    dropout_scale = tl.full((BLOCK_R, BLOCK_C), DROPOUT_SCALE_, tl.float32)

    acc_x_norm = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = _add_rn(_add_rn(x0, x1), x2)
        norm = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        weighted = _mul_rn(x, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, norm), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(norm, row_dot[:, None]))
        ln_grad = _mul_rn(scale[:, None], centered)

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        keep_scale = _mul_rn(keep, dropout_scale)
        producer = _mul_rn(_add_rn(residual, ln_grad), keep_scale)
        tl.store(producer_tmp_ptr + offsets, producer, mask=mask)

        acc_x_norm += tl.sum(tl.where(mask, _mul_rn(x, norm), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)

        seq = rows - (rows // SEQ_) * SEQ_
        position_raw = tl.load(position_idx_ptr + seq, mask=row_mask, other=-1).to(tl.int64)
        position_valid = (
            (position_raw[:, None] >= 0)
            & (position_raw[:, None] < POSITION_ROWS_)
            & (position_raw[:, None] != -1)
        )
        position_row = tl.where(position_valid, position_raw[:, None], 0)
        tl.atomic_add(
            position_out_ptr + position_row * HIDDEN_ + cols[None, :],
            producer,
            sem="relaxed",
            mask=mask & position_valid,
        )

        vocab_raw = tl.load(vocab_idx_ptr + rows, mask=row_mask, other=0).to(tl.int64)
        vocab_valid = (
            (vocab_raw[:, None] >= 0)
            & (vocab_raw[:, None] < VOCAB_ROWS_)
            & (vocab_raw[:, None] != 0)
        )
        vocab_row = tl.where(vocab_valid, vocab_raw[:, None], 0)
        tl.atomic_add(
            vocab_out_ptr + vocab_row * HIDDEN_ + cols[None, :],
            producer,
            sem="relaxed",
            mask=mask & vocab_valid,
        )

    partial_base = group * 2 * HIDDEN_ + cols
    tl.store(partials_ptr + partial_base, acc_x_norm, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN_, acc_x, mask=col_mask)


@triton.jit
def _segment_chunk_kernel(
    producer_tmp_ptr,
    segment_idx_ptr,
    segment_partials_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    ROWS_PER_CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    chunk = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local in tl.range(0, ROWS_PER_CHUNK):
        row = chunk * ROWS_PER_CHUNK + local
        segment_raw = tl.load(segment_idx_ptr + row).to(tl.int64)
        segment_row = tl.maximum(tl.minimum(segment_raw, SEGMENT_ROWS_ - 1), 0)
        values = tl.load(
            producer_tmp_ptr + row * HIDDEN_ + cols,
            mask=(row < ROWS_) & col_mask,
            other=0.0,
        ).to(tl.float32)
        acc0 = _add_rn(acc0, tl.where((row < ROWS_) & (segment_row == 0), values, 0.0))
        acc1 = _add_rn(acc1, tl.where((row < ROWS_) & (segment_row == 1), values, 0.0))

    partial_base = chunk * 2 * HIDDEN_ + cols
    tl.store(segment_partials_ptr + partial_base, acc0, mask=col_mask)
    tl.store(segment_partials_ptr + partial_base + HIDDEN_, acc1, mask=col_mask)


@triton.jit
def _segment_finalize_kernel(
    segment_partials_ptr,
    segment_out_ptr,
    NUM_CHUNKS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for chunk in tl.range(0, NUM_CHUNKS):
        offsets = chunk * 2 * HIDDEN_ + cols
        part0 = tl.load(segment_partials_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        part1 = tl.load(segment_partials_ptr + offsets + HIDDEN_, mask=col_mask, other=0.0).to(tl.float32)
        acc0 = _add_rn(acc0, part0)
        acc1 = _add_rn(acc1, part1)

    tl.store(segment_out_ptr + cols, acc0, mask=col_mask)
    tl.store(segment_out_ptr + HIDDEN_ + cols, acc1, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 2 * HIDDEN_ + cols[None, :]

    x_norm = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_norm_ptr + cols, tl.sum(x_norm, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="8ab09cfc",
    ROWS_PER_GROUP=4,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=8,
    INIT_BLOCK=1024,
    SEGMENT_ROWS_PER_CHUNK=256,
    SEGMENT_BLOCK_C=16,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    INIT_BLOCK: int,
    SEGMENT_ROWS_PER_CHUNK: int,
    SEGMENT_BLOCK_C: int,
    row_warps: int,
    final_warps: int,
):
    (
        base_vocab,
        x0,
        x1,
        x2,
        weight,
        norm,
        scale,
        residual,
        keep,
        position_idx,
        segment_idx,
        vocab_idx,
        _shape0,
        _shape1,
        _shape2,
        position_shape_param,
        _mask_shape_param,
        segment_shape_param,
        vocab_shape_param,
    ) = inputs

    device = base_vocab.device
    position_out = torch.empty_strided(
        _shape_tuple(position_shape_param),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    segment_out = torch.empty_strided(
        _shape_tuple(segment_shape_param),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    vocab_out = torch.empty_strided(
        _shape_tuple(vocab_shape_param),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    out_x_norm = torch.empty_strided((HIDDEN,), (1,), device=device, dtype=torch.float32)
    out_x = torch.empty_strided((HIDDEN,), (1,), device=device, dtype=torch.float32)

    total_position = POSITION_ROWS * HIDDEN
    total_segment = SEGMENT_ROWS * HIDDEN
    total_vocab = VOCAB_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        base_vocab,
        position_out,
        segment_out,
        vocab_out,
        TOTAL_POSITION=total_position,
        TOTAL_SEGMENT=total_segment,
        TOTAL_VOCAB=total_vocab,
        BLOCK=INIT_BLOCK,
        num_warps=4,
    )

    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 2, HIDDEN),
        (2 * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    producer_tmp = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    num_segment_chunks = triton.cdiv(ROWS, SEGMENT_ROWS_PER_CHUNK)
    segment_partials = torch.empty_strided(
        (num_segment_chunks, 2, HIDDEN),
        (2 * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    _row_partials_scatter_kernel[(num_groups,)](
        x0,
        x1,
        x2,
        weight,
        norm,
        scale,
        residual,
        keep,
        position_idx,
        segment_idx,
        vocab_idx,
        position_out,
        segment_out,
        vocab_out,
        partials,
        producer_tmp,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        num_warps=row_warps,
    )

    _segment_chunk_kernel[(num_segment_chunks, triton.cdiv(HIDDEN, SEGMENT_BLOCK_C))](
        producer_tmp,
        segment_idx,
        segment_partials,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        ROWS_PER_CHUNK=SEGMENT_ROWS_PER_CHUNK,
        BLOCK_C=SEGMENT_BLOCK_C,
        num_warps=4,
    )
    _segment_finalize_kernel[(triton.cdiv(HIDDEN, SEGMENT_BLOCK_C),)](
        segment_partials,
        segment_out,
        NUM_CHUNKS=num_segment_chunks,
        HIDDEN_=HIDDEN,
        BLOCK_C=SEGMENT_BLOCK_C,
        num_warps=4,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_C),)](
        partials,
        out_x_norm,
        out_x,
        NUM_GROUPS=num_groups,
        HIDDEN_=HIDDEN,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    return out_x_norm, out_x, position_out, segment_out, vocab_out
