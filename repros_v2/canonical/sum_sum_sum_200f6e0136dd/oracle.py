"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete BERT bf16 layer-norm-backward/dropout embedding-gradient tuple by reconstructing the three-bf16-input f32 producer with the captured division/add/dropout ordering, accumulating the two returned hidden-column reductions from cooperative row partials, and directly applying the two duplicate-preserving masked `_unsafe_masked_index_put_accumulate` scatter-add outputs into `[3,768]` and `[20005,768]`, whereas Inductor materializes the rowwise producer and lowers the sibling reductions, validity masks, zero fills, and indexed accumulates as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no scatter-reduce template that keeps row-local layer-norm-backward reduction scalars live while feeding multiple masked embedding-gradient accumulators and hidden reductions; the fix is SCATTER_REDUCE: add a guarded BERT embedding-gradient scatter-reduce lowering that folds validity masks into indexed updates and finalizes compatible hidden-column reductions from shared row partials while preserving bf16-to-f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 2048
CHANNELS = 768
SEGMENT_ROWS = 3
VOCAB_ROWS = 20005
EPS = 1.0e-6
ROW_BACKWARD_SCALE = 0.002607561929595828
RETURN_DROPOUT_SCALE = 1.1111111111111112


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
def _div_rn(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _zero_outputs_kernel(
    segment_out_ptr,
    vocab_out_ptr,
    SEGMENT_TOTAL: tl.constexpr,
    VOCAB_TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(vocab_out_ptr + offsets, zeros, mask=offsets < VOCAB_TOTAL)
    tl.store(segment_out_ptr + offsets, zeros, mask=offsets < SEGMENT_TOTAL)


@triton.jit
def _row_partials_scatter_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    gamma_ptr,
    dy_ptr,
    denom_base_ptr,
    residual_ptr,
    full_scalar_ptr,
    keep_ptr,
    segment_idx_ptr,
    vocab_idx_ptr,
    segment_out_ptr,
    vocab_out_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EPS_: tl.constexpr,
    ROW_BACKWARD_SCALE_: tl.constexpr,
    RETURN_DROPOUT_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS_
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    full_scalar = tl.load(full_scalar_ptr + 0).to(tl.float32)

    acc_sum_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_sum_x_dy_over_denom = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS_
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * CHANNELS_ + cols[None, :]

        a0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a1 = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a2 = tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = _add_rn(_add_rn(a0, a1), a2)

        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        denom_base = tl.load(denom_base_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        denom = _add_rn(denom_base, tl.full((BLOCK_R,), EPS_, tl.float32))

        gamma_dy = _mul_rn(gamma[None, :], dy)
        gamma_dy_over_denom = _div_rn(gamma_dy, denom[:, None])
        gamma_dy_over_denom2 = _div_rn(gamma_dy_over_denom, denom[:, None])
        sum2_terms = _mul_rn(-x, gamma_dy_over_denom2)
        row_sum2 = tl.sum(tl.where(mask, sum2_terms, 0.0), axis=1)

        x_over_denom = _div_rn(x, denom[:, None])
        x_gamma_over_denom = _mul_rn(x_over_denom, gamma[None, :])
        x_dy_over_denom = _mul_rn(x_over_denom, dy)

        row_sum4 = -tl.sum(tl.where(mask, x_gamma_over_denom, 0.0), axis=1)
        denom_twice = _mul_rn(denom_base, tl.full((BLOCK_R,), 2.0, tl.float32))
        div3 = _div_rn(row_sum2, denom_twice)
        where_val = tl.where(denom_base == 0.0, full_scalar, div3)
        row_coef = _mul_rn(where_val, tl.full((BLOCK_R,), ROW_BACKWARD_SCALE_, tl.float32))

        add3 = _add_rn(residual, x_gamma_over_denom)
        add4 = _add_rn(add3, _mul_rn(row_coef[:, None], dy))
        div4 = _div_rn(row_sum4, tl.full((BLOCK_R,), 768.0, tl.float32))
        add5 = _add_rn(add4, div4[:, None])

        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        keep_scale = _mul_rn(
            keep,
            tl.full((BLOCK_R, BLOCK_C), RETURN_DROPOUT_SCALE_, tl.float32),
        )
        producer = _mul_rn(add5, keep_scale)

        acc_sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_sum_x_dy_over_denom += tl.sum(tl.where(mask, x_dy_over_denom, 0.0), axis=0)

        segment_raw = tl.load(segment_idx_ptr + rows, mask=row_mask, other=0).to(tl.int64)
        segment_valid = (
            (segment_raw[:, None] >= 0)
            & (segment_raw[:, None] < SEGMENT_ROWS_)
            & (segment_raw[:, None] != 0)
        )
        segment_row = tl.where(segment_valid, segment_raw[:, None], 0)
        tl.atomic_add(
            segment_out_ptr + segment_row * CHANNELS_ + cols[None, :],
            producer,
            sem="relaxed",
            mask=mask & segment_valid,
        )

        vocab_raw = tl.load(vocab_idx_ptr + rows, mask=row_mask, other=0).to(tl.int64)
        vocab_valid = (
            (vocab_raw[:, None] >= 0)
            & (vocab_raw[:, None] < VOCAB_ROWS_)
            & (vocab_raw[:, None] != 0)
        )
        vocab_row = tl.where(vocab_valid, vocab_raw[:, None], 0)
        tl.atomic_add(
            vocab_out_ptr + vocab_row * CHANNELS_ + cols[None, :],
            producer,
            sem="relaxed",
            mask=mask & vocab_valid,
        )

    partial_base = group * 2 * CHANNELS_ + cols
    tl.store(partials_ptr + partial_base, acc_sum_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + CHANNELS_, acc_sum_x_dy_over_denom, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    sum_x_ptr,
    sum_x_dy_over_denom_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < CHANNELS_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 2 * CHANNELS_ + cols[None, :]

    sum_x = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x_dy = tl.load(partials_ptr + offsets + CHANNELS_, mask=mask, other=0.0).to(tl.float32)

    tl.store(sum_x_ptr + cols, tl.sum(sum_x, axis=0), mask=col_mask)
    tl.store(sum_x_dy_over_denom_ptr + cols, tl.sum(sum_x_dy, axis=0), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="3d639bb6",
    ROWS_PER_GROUP=4,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    INIT_BLOCK=1024,
    num_warps=8,
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
    num_warps: int,
    final_warps: int,
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
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        shape_sum0,
        shape_sum1,
        _shape_param_5,
        shape_segment,
        shape_vocab,
    ) = inputs

    device = arg0_1.device
    sum0 = torch.empty_strided(_shape_tuple(shape_sum0), (1,), device=device, dtype=torch.float32)
    sum1 = torch.empty_strided(_shape_tuple(shape_sum1), (1,), device=device, dtype=torch.float32)
    segment_out = torch.empty(_shape_tuple(shape_segment), device=device, dtype=torch.float32)
    vocab_out = torch.empty(_shape_tuple(shape_vocab), device=device, dtype=torch.float32)

    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    partials = torch.empty((num_groups, 2, CHANNELS), device=device, dtype=torch.float32)

    segment_total = SEGMENT_ROWS * CHANNELS
    vocab_total = VOCAB_ROWS * CHANNELS
    _zero_outputs_kernel[(triton.cdiv(vocab_total, INIT_BLOCK),)](
        segment_out,
        vocab_out,
        SEGMENT_TOTAL=segment_total,
        VOCAB_TOTAL=vocab_total,
        BLOCK=INIT_BLOCK,
        num_warps=4,
    )

    _row_partials_scatter_kernel[(num_groups,)](
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
        segment_out,
        vocab_out,
        partials,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        EPS_=EPS,
        ROW_BACKWARD_SCALE_=ROW_BACKWARD_SCALE,
        RETURN_DROPOUT_SCALE_=RETURN_DROPOUT_SCALE,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partials,
        sum0,
        sum1,
        NUM_GROUPS=num_groups,
        CHANNELS_=CHANNELS,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    return sum0, sum1, segment_out, vocab_out
