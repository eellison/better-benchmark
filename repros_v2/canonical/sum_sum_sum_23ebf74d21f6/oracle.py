"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BERT bf16 layer-norm-backward/dropout return tuple by row-tiling the three-bf16-input producer, sharing each row's hidden-dimension reductions, writing the returned f32 gradient tensor plus the bf16 flat/transpose side output, and cooperatively finalizing the three returned hidden-column reductions from common row partials, whereas Inductor schedules the add producer, row reductions, dropout epilogues, bf16 materialization, view fanout, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that preserves explicit bf16 rounding boundaries while coordinating row-local scalars, side-output stores, and compatible column partials; the fix is COOPERATIVE_SPLIT_K: add a guarded BERT layer-norm-backward row-tiled reduction plan that keeps row summaries live, sinks view-only layout algebra into output planning, and finalizes sibling column reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 2048
CHANNELS = 768
EPS = 1.0e-6
ROW_BACKWARD_SCALE = 0.002607561929595828
RETURN_DROPOUT_SCALE = 1.1111111111111112
BF16_DROPOUT_SCALE = 1.109375


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
def _row_partials_and_outputs_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    gamma_ptr,
    dy_ptr,
    denom_base_ptr,
    residual_ptr,
    full_scalar_ptr,
    keep0_ptr,
    keep1_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    EPS_: tl.constexpr,
    ROW_BACKWARD_SCALE_: tl.constexpr,
    RETURN_DROPOUT_SCALE_: tl.constexpr,
    BF16_DROPOUT_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS_
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    full_scalar = tl.load(full_scalar_ptr + 0).to(tl.float32)

    acc_sum_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_sum_x_dy_over_denom = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_sum_bf16_out = tl.zeros((BLOCK_C,), dtype=tl.float32)

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
        denom_base = tl.load(denom_base_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )
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
        row_coef = _mul_rn(
            where_val,
            tl.full((BLOCK_R,), ROW_BACKWARD_SCALE_, tl.float32),
        )

        add3 = _add_rn(residual, x_gamma_over_denom)
        add4 = _add_rn(add3, _mul_rn(row_coef[:, None], dy))
        div4 = _div_rn(
            row_sum4,
            tl.full((BLOCK_R,), 768.0, tl.float32),
        )
        add5 = _add_rn(add4, div4[:, None])

        keep0 = tl.load(keep0_ptr + offsets, mask=mask, other=0).to(tl.float32)
        keep0_scale = _mul_rn(
            keep0,
            tl.full((BLOCK_R, BLOCK_C), RETURN_DROPOUT_SCALE_, tl.float32),
        )
        out_f32 = _mul_rn(add5, keep0_scale)

        keep1 = tl.load(keep1_ptr + offsets, mask=mask, other=0).to(tl.float32)
        keep1_scale = _mul_rn(
            keep1,
            tl.full((BLOCK_R, BLOCK_C), BF16_DROPOUT_SCALE_, tl.float32),
        ).to(tl.bfloat16).to(tl.float32)
        out_bf16 = _mul_rn(out_f32.to(tl.bfloat16).to(tl.float32), keep1_scale).to(
            tl.bfloat16
        )

        tl.store(out_f32_ptr + offsets, out_f32, mask=mask)
        tl.store(out_bf16_ptr + offsets, out_bf16, mask=mask)

        acc_sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_sum_x_dy_over_denom += tl.sum(
            tl.where(mask, x_dy_over_denom, 0.0),
            axis=0,
        )
        acc_sum_bf16_out += tl.sum(
            tl.where(mask, out_bf16.to(tl.float32), 0.0),
            axis=0,
        )

    partial_base = group * 3 * CHANNELS_ + cols
    tl.store(partials_ptr + partial_base, acc_sum_x, mask=col_mask)
    tl.store(
        partials_ptr + partial_base + CHANNELS_,
        acc_sum_x_dy_over_denom,
        mask=col_mask,
    )
    tl.store(
        partials_ptr + partial_base + 2 * CHANNELS_,
        acc_sum_bf16_out,
        mask=col_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    sum_x_ptr,
    sum_x_dy_over_denom_ptr,
    sum_bf16_out_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < CHANNELS_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * CHANNELS_ + cols[None, :]

    sum_x = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x_dy = tl.load(partials_ptr + offsets + CHANNELS_, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_bf16 = tl.load(
        partials_ptr + offsets + 2 * CHANNELS_,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    tl.store(sum_x_ptr + cols, tl.sum(sum_x, axis=0), mask=col_mask)
    tl.store(sum_x_dy_over_denom_ptr + cols, tl.sum(sum_x_dy, axis=0), mask=col_mask)
    tl.store(
        sum_bf16_out_ptr + cols,
        tl.sum(sum_bf16, axis=0).to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 6b5bc342: BERT_pytorch train, [2048, 768] bf16 three-input LN-backward/dropout tail.
@oracle_impl(
    hardware="B200",
    point="6b5bc342",
    ROWS_PER_GROUP=4,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
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
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        shape_sum0,
        shape_sum1,
        shape_out_f32,
        shape_out_bf16,
        shape_sum_bf16,
    ) = inputs

    device = arg0_1.device
    out_f32_shape = _shape_tuple(shape_out_f32)
    out_bf16_shape = _shape_tuple(shape_out_bf16)
    sum0_shape = _shape_tuple(shape_sum0)
    sum1_shape = _shape_tuple(shape_sum1)
    sum_bf16_shape = _shape_tuple(shape_sum_bf16)

    out_f32 = torch.empty_strided(
        out_f32_shape,
        (out_f32_shape[1] * out_f32_shape[2], out_f32_shape[2], 1),
        device=device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        out_bf16_shape,
        (out_bf16_shape[1], 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_x = torch.empty_strided(sum0_shape, (1,), device=device, dtype=torch.float32)
    sum_x_dy = torch.empty_strided(sum1_shape, (1,), device=device, dtype=torch.float32)
    sum_bf16 = torch.empty_strided(
        sum_bf16_shape,
        (1,),
        device=device,
        dtype=torch.float32,
    )

    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, CHANNELS),
        (3 * CHANNELS, CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )

    _row_partials_and_outputs_kernel[(num_groups,)](
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
        out_f32,
        out_bf16,
        partials,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        EPS_=EPS,
        ROW_BACKWARD_SCALE_=ROW_BACKWARD_SCALE,
        RETURN_DROPOUT_SCALE_=RETURN_DROPOUT_SCALE,
        BF16_DROPOUT_SCALE_=BF16_DROPOUT_SCALE,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partials,
        sum_x,
        sum_x_dy,
        sum_bf16,
        NUM_GROUPS=num_groups,
        CHANNELS_=CHANNELS,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    return sum_x, sum_x_dy, out_f32, out_bf16, out_bf16.permute(1, 0), sum_bf16
