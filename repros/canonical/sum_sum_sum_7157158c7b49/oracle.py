"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete ConvBERT bf16 residual/dropout layer-norm-backward plus embedding-gradient scatter scope by reconstructing the captured bf16-to-f32 add chain and dropout mask inside one row-local Triton producer, emitting both `[768]` column reductions, batch-reducing the position-gradient producer, and directly applying the token-type and vocabulary duplicate-preserving masked scatter-adds while copying the bf16 vocabulary base into the returned f32 table, whereas Inductor materializes the rowwise producer and lowers the sibling reductions, mask construction, three `_unsafe_masked_index_put_accumulate` destinations, and dense vocabulary add as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no scatter-reduce template that fuses a row-local layer-norm-backward producer with multiple masked indexed accumulators while preserving bf16 cast boundaries, full dense output scope, and duplicate-safe atomics; the fix is SCATTER_REDUCE: add a guarded embedding-gradient scatter-reduce lowering that folds validity masks into indexed updates, initializes dense destinations directly, and finalizes compatible hidden-column reductions from cooperative row partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
HIDDEN = 768
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 30522
DROP_SCALE = 1.1111111111111112


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
    segment_out_ptr,
    position_out_ptr,
    vocab_out_ptr,
    VOCAB_N: tl.constexpr,
    SEGMENT_N: tl.constexpr,
    POSITION_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)

    vocab_mask = offsets < VOCAB_N
    base = tl.load(base_vocab_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(vocab_out_ptr + offsets, base, mask=vocab_mask)
    tl.store(segment_out_ptr + offsets, zeros, mask=offsets < SEGMENT_N)
    tl.store(position_out_ptr + offsets, zeros, mask=offsets < POSITION_N)


@triton.jit
def _seq_scatter_partials_kernel(
    bf16_a_ptr,
    residual_ptr,
    bf16_b_ptr,
    bf16_permuted_ptr,
    bf16_c_ptr,
    bf16_d_ptr,
    keep_ptr,
    weight_ptr,
    normed_ptr,
    row_scale_ptr,
    segment_index_ptr,
    position_index_ptr,
    vocab_index_ptr,
    partial_x_normed_ptr,
    partial_x_ptr,
    segment_out_ptr,
    position_out_ptr,
    vocab_out_ptr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seq = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_normed = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_position = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for batch in tl.static_range(0, BATCH_):
        row = batch * SEQ_ + seq
        flat_offsets = row * HIDDEN_ + cols
        perm_offsets = batch * (HIDDEN_ * SEQ_) + cols * SEQ_ + seq
        mask = col_mask

        residual_val = tl.load(residual_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        added = _add_rn(
            residual_val,
            tl.load(bf16_a_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32),
        )
        added = _add_rn(
            added,
            tl.load(bf16_b_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32),
        )
        added = _add_rn(
            added,
            tl.load(bf16_permuted_ptr + perm_offsets, mask=mask, other=0.0).to(tl.float32),
        )
        added = _add_rn(
            added,
            tl.load(bf16_c_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32),
        )
        added = _add_rn(
            added,
            tl.load(bf16_d_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32),
        )

        keep = tl.load(keep_ptr + flat_offsets, mask=mask, other=0).to(tl.float32)
        drop = _mul_rn(keep, DROP_SCALE_)
        x = _mul_rn(added, drop)

        normed = tl.load(normed_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = _mul_rn(x, weight)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, normed), 0.0), axis=0)
        scaled = _mul_rn(weighted, 768.0)
        centered = _sub_rn(_sub_rn(scaled, row_sum), _mul_rn(normed, row_dot))
        row_scale = tl.load(row_scale_ptr + row, mask=True, other=0.0).to(tl.float32)
        producer = _mul_rn(row_scale, centered)

        acc_x_normed += tl.where(mask, _mul_rn(x, normed), 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_position += tl.where(mask, producer, 0.0)

        segment_raw = tl.load(segment_index_ptr + seq).to(tl.int64)
        segment_valid = (
            col_mask
            & (segment_raw >= 0)
            & (segment_raw < SEGMENT_ROWS_)
            & (segment_raw != -1)
        )
        tl.atomic_add(
            segment_out_ptr + segment_raw * HIDDEN_ + cols,
            producer,
            sem="relaxed",
            mask=segment_valid,
        )

        vocab_raw = tl.load(vocab_index_ptr + row).to(tl.int64)
        vocab_valid = (
            col_mask
            & (vocab_raw >= 0)
            & (vocab_raw < VOCAB_ROWS_)
            & (vocab_raw != 0)
        )
        tl.atomic_add(
            vocab_out_ptr + vocab_raw * HIDDEN_ + cols,
            producer,
            sem="relaxed",
            mask=vocab_valid,
        )

    position_raw = tl.load(position_index_ptr + seq).to(tl.int64)
    position_valid = (
        col_mask
        & (position_raw >= 0)
        & (position_raw < POSITION_ROWS_)
        & (position_raw != -1)
    )
    tl.atomic_add(
        position_out_ptr + position_raw * HIDDEN_ + cols,
        acc_position,
        sem="relaxed",
        mask=position_valid,
    )

    partial_offsets = seq * HIDDEN_ + cols
    tl.store(partial_x_normed_ptr + partial_offsets, acc_x_normed, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_x_normed_ptr,
    partial_x_ptr,
    out_x_normed_ptr,
    out_x_ptr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    seqs = tl.arange(0, BLOCK_SEQ)
    mask = (seqs[:, None] < SEQ_) & (cols[None, :] < HIDDEN_)
    offsets = seqs[:, None] * HIDDEN_ + cols[None, :]

    x_normed = tl.load(partial_x_normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    col_mask = cols < HIDDEN_
    tl.store(out_x_normed_ptr + cols, tl.sum(x_normed, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="50c7603a",
    INIT_BLOCK=1024,
    BLOCK_H=1024,
    FINAL_BLOCK_H=8,
    num_warps_init=4,
    num_warps_seq=8,
    num_warps_final=8,
)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
    num_warps_init: int,
    num_warps_seq: int,
    num_warps_final: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    partial_x_normed = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_x = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out0 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    segment_out = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    vocab_n = VOCAB_ROWS * HIDDEN
    segment_n = SEGMENT_ROWS * HIDDEN
    position_n = POSITION_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(vocab_n, INIT_BLOCK),)](
        arg0_1,
        segment_out,
        position_out,
        vocab_out,
        VOCAB_N=vocab_n,
        SEGMENT_N=segment_n,
        POSITION_N=position_n,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _seq_scatter_partials_kernel[(SEQ,)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        partial_x_normed,
        partial_x,
        segment_out,
        position_out,
        vocab_out,
        BATCH_=BATCH,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROP_SCALE_=DROP_SCALE,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_seq,
    )

    _finalize_partials_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_H),)](
        partial_x_normed,
        partial_x,
        out0,
        out1,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_SEQ=_ceil_pow2(SEQ),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=num_warps_final,
    )

    return out0, out1, segment_out, position_out, vocab_out
