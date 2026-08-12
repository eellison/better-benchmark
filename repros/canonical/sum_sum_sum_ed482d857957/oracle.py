"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GoogleFnet layer-norm-backward plus embedding-gradient scatter scope by keeping each row's hidden-dimension reductions in one Triton producer, emitting the two `[768]` column-sum partial streams, and directly applying the three duplicate-preserving masked scatter-adds into the returned `[512,768]`, `[4,768]`, and `arg7 + [32000,768]` outputs, whereas Inductor lowers the row reductions, sibling column reductions, mask construction, zero fills, indexed accumulates, and dense add as separate generic reduction/scatter regions around materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no scatter-reduce template that fuses a row-local layer-norm-backward producer with multiple masked index_put-accumulate destinations while preserving the full dense output scope and duplicate-safe atomic accumulation; the fix is SCATTER_REDUCE: add a guarded lowering for masked embedding-gradient scatter-reduces that folds validity masks into the indexed updates, initializes dense destinations directly, and finalizes compatible column reductions from cooperative row partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
POSITION_ROWS = 512
SEGMENT_ROWS = 4
VOCAB_ROWS = 32000


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
    pos_out_ptr,
    seg_out_ptr,
    vocab_out_ptr,
    VOCAB_N: tl.constexpr,
    POS_N: tl.constexpr,
    SEG_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    vocab_mask = offsets < VOCAB_N
    values = tl.load(base_vocab_ptr + offsets, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(vocab_out_ptr + offsets, values, mask=vocab_mask)
    tl.store(pos_out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < POS_N)
    tl.store(seg_out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < SEG_N)


@triton.jit
def _row_scatter_partials_kernel(
    x_ptr,
    weight_ptr,
    normed_ptr,
    row_scale_ptr,
    pos_index_ptr,
    segment_index_ptr,
    vocab_index_ptr,
    partial_x_normed_ptr,
    partial_x_ptr,
    pos_out_ptr,
    segment_out_ptr,
    vocab_out_ptr,
    ROWS_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_normed = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for row_offset in tl.static_range(0, ROWS_PER_GROUP):
        row = group * ROWS_PER_GROUP + row_offset
        row_active = row < ROWS_
        token = row % SEQ_
        offsets = row * HIDDEN_ + cols
        mask = row_active & col_mask

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row_active, other=0.0).to(tl.float32)

        weighted = _mul_rn(x, weight)
        scaled_weighted = _mul_rn(weighted, ROW_FACTOR_)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        weighted_normed = _mul_rn(weighted, normed)
        row_dot = tl.sum(tl.where(mask, weighted_normed, 0.0), axis=0)
        centered = _sub_rn(_sub_rn(scaled_weighted, row_sum), _mul_rn(normed, row_dot))
        grad = _mul_rn(row_scale, centered)

        acc_x_normed += tl.where(mask, _mul_rn(x, normed), 0.0)
        acc_x += tl.where(mask, x, 0.0)

        pos_row = tl.load(pos_index_ptr + token, mask=row_active, other=-1).to(tl.int64)
        pos_active = (
            mask
            & (pos_row >= 0)
            & (pos_row < POSITION_ROWS_)
            & (pos_row != -1)
        )
        tl.atomic_add(
            pos_out_ptr + pos_row * HIDDEN_ + cols,
            grad,
            sem="relaxed",
            mask=pos_active,
        )

        segment_row = tl.load(segment_index_ptr + token, mask=row_active, other=-1).to(tl.int64)
        segment_active = (
            mask
            & (segment_row >= 0)
            & (segment_row < SEGMENT_ROWS_)
            & (segment_row != -1)
        )
        tl.atomic_add(
            segment_out_ptr + segment_row * HIDDEN_ + cols,
            grad,
            sem="relaxed",
            mask=segment_active,
        )

        vocab_row = tl.load(vocab_index_ptr + row, mask=row_active, other=3).to(tl.int64)
        vocab_active = (
            mask
            & (vocab_row >= 0)
            & (vocab_row < VOCAB_ROWS_)
            & (vocab_row != 3)
        )
        tl.atomic_add(
            vocab_out_ptr + vocab_row * HIDDEN_ + cols,
            grad,
            sem="relaxed",
            mask=vocab_active,
        )

    partial_offsets = group * HIDDEN_ + cols
    tl.store(partial_x_normed_ptr + partial_offsets, acc_x_normed, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_x_normed_ptr,
    partial_x_ptr,
    out_x_normed_ptr,
    out_x_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * HIDDEN_ + cols[None, :]

    x_normed = tl.load(partial_x_normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_normed_ptr + cols, tl.sum(x_normed, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="f69e9b69",
    ROWS_PER_GROUP=8,
    BLOCK_H=1024,
    FINAL_BLOCK_H=8,
    INIT_BLOCK=1024,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
    INIT_BLOCK: int,
    row_warps: int,
    final_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        *_shape_params,
    ) = inputs

    device = arg0.device
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])
    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)

    partial_x_normed = torch.empty((num_groups, hidden), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_groups, hidden), device=device, dtype=torch.float32)
    out_x_normed = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_x = torch.empty((hidden,), device=device, dtype=torch.float32)
    pos_out = torch.empty((POSITION_ROWS, hidden), device=device, dtype=torch.float32)
    segment_out = torch.empty((SEGMENT_ROWS, hidden), device=device, dtype=torch.float32)
    vocab_out = torch.empty_strided(
        tuple(int(dim) for dim in arg7.shape),
        tuple(int(stride) for stride in arg7.stride()),
        device=device,
        dtype=torch.float32,
    )

    vocab_n = int(arg7.numel())
    pos_n = POSITION_ROWS * hidden
    segment_n = SEGMENT_ROWS * hidden
    _init_outputs_kernel[(triton.cdiv(vocab_n, INIT_BLOCK),)](
        arg7,
        pos_out,
        segment_out,
        vocab_out,
        VOCAB_N=vocab_n,
        POS_N=pos_n,
        SEG_N=segment_n,
        BLOCK=INIT_BLOCK,
        num_warps=4,
    )
    _row_scatter_partials_kernel[(num_groups,)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        partial_x_normed,
        partial_x,
        pos_out,
        segment_out,
        vocab_out,
        ROWS_=rows,
        SEQ_=SEQ,
        HIDDEN_=hidden,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_H=BLOCK_H,
        ROW_FACTOR_=float(HIDDEN),
        num_warps=row_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partial_x_normed,
        partial_x,
        out_x_normed,
        out_x,
        NUM_GROUPS=num_groups,
        HIDDEN_=hidden,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=final_warps,
    )

    return out_x_normed, out_x, pos_out, segment_out, vocab_out
