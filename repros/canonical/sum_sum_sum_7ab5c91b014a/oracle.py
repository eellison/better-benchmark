"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BERT bf16 layer-norm-backward-style return tuple by row-tiling the `[2048, 768]` producer, keeping row-local hidden reductions live through the fp32 dense epilogue, materializing the returned bf16 dropout buffer and its transpose alias with the explicit bf16 cast/product boundaries, and cooperatively finalizing the two fp32 column sums plus the bf16-rounded dropout column sum, whereas Inductor lowers the row reductions, dense pointwise store, bf16 dropout materialization, transpose view, and sibling column reductions as separate generic schedules over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that coordinates row-local layer-norm-backward scalars, dependent returned tensor stores, and multiple compatible column reductions while preserving dtype boundaries and aliases; the fix is COOPERATIVE_SPLIT_K: add a guarded row-tiled template that writes the observable dense outputs once and finalizes all sibling column partials directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-6
ROW_BACKWARD_SCALE = 0.002607561929595828
DROPOUT_SCALE = 1.1111111111111112
INV_HIDDEN = 1.0 / 768.0


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
def _row_partials_kernel(
    x_ptr,
    gamma_ptr,
    dy_ptr,
    denom_base_ptr,
    residual_ptr,
    keep_ptr,
    grad_out_ptr,
    drop_out_ptr,
    zero_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EPS_: tl.constexpr,
    INV_HIDDEN_: tl.constexpr,
    ROW_BACKWARD_SCALE_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_mul3 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_drop = tl.zeros((BLOCK_H,), dtype=tl.float32)

    tl.store(zero_ptr, 0.0, mask=group == 0)

    for row_inner in tl.static_range(0, ROWS_PER_GROUP):
        row = group * ROWS_PER_GROUP + row_inner
        row_active = row < ROWS
        offsets = row * HIDDEN + cols
        mask = row_active & col_mask

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        denom_base = tl.load(denom_base_ptr + row, mask=row_active, other=0.0).to(tl.float32)
        denom = _add_rn(denom_base, EPS_)

        mul = _mul_rn(gamma, dy)
        div = _div_rn(mul, denom)
        div_1 = _div_rn(div, denom)
        mul_1 = _mul_rn(-x, div_1)
        sum_2 = tl.sum(tl.where(mask, mul_1, 0.0), axis=0)

        div_2 = _div_rn(x, denom)
        mul_2 = _mul_rn(div_2, gamma)
        mul_3 = _mul_rn(div_2, dy)
        sum_4 = tl.sum(tl.where(mask, -mul_2, 0.0), axis=0)

        add_1 = _add_rn(residual, mul_2)
        mul_4 = _mul_rn(denom_base, 2.0)
        div_3 = _div_rn(sum_2, mul_4)
        where = tl.where(denom_base == 0.0, 0.0, div_3)
        mul_5 = _mul_rn(where, ROW_BACKWARD_SCALE_)
        mul_6 = _mul_rn(mul_5, dy)
        add_2 = _add_rn(add_1, mul_6)
        div_4 = _mul_rn(sum_4, INV_HIDDEN_)
        add_3 = _add_rn(add_2, div_4)

        tl.store(grad_out_ptr + offsets, add_3, mask=mask)

        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        add_3_bf16 = add_3.to(tl.bfloat16)
        keep_scaled_bf16 = _mul_rn(keep, DROPOUT_SCALE_).to(tl.bfloat16)
        drop_bf16 = _mul_rn(
            add_3_bf16.to(tl.float32),
            keep_scaled_bf16.to(tl.float32),
        ).to(tl.bfloat16)
        tl.store(drop_out_ptr + offsets, drop_bf16, mask=mask)

        acc_x += tl.where(mask, x, 0.0)
        acc_mul3 += tl.where(mask, mul_3, 0.0)
        acc_drop += tl.where(mask, drop_bf16.to(tl.float32), 0.0)

    partial_base = group * 3 * HIDDEN + cols
    tl.store(partials_ptr + partial_base, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN, acc_mul3, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN, acc_drop, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    sum_x_ptr,
    sum_mul3_ptr,
    sum_drop_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * HIDDEN + cols[None, :]

    x = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mul3 = tl.load(partials_ptr + offsets + HIDDEN, mask=mask, other=0.0).to(tl.float32)
    drop = tl.load(partials_ptr + offsets + 2 * HIDDEN, mask=mask, other=0.0).to(tl.float32)

    tl.store(sum_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(sum_mul3_ptr + cols, tl.sum(mul3, axis=0), mask=col_mask)
    rounded_drop = tl.sum(drop, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(sum_drop_ptr + cols, rounded_drop, mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="801fb66e",
    ROWS_PER_GROUP=4,
    BLOCK_H=1024,
    FINAL_BLOCK_H=8,
    row_warps=4,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
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
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])
    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    device = arg0.device

    add_3 = torch.empty_strided(
        _shape_tuple(shape3),
        (int(shape0[1]) * hidden, hidden, 1),
        device=device,
        dtype=torch.float32,
    )
    view_3 = torch.empty_strided(
        _shape_tuple(shape4),
        (hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    view_1 = torch.empty_strided(_shape_tuple(shape1), (1,), device=device, dtype=torch.float32)
    view_2 = torch.empty_strided(_shape_tuple(shape2), (1,), device=device, dtype=torch.float32)
    full = torch.empty((), device=device, dtype=torch.float32)
    convert_element_type_4 = torch.empty_strided(
        _shape_tuple(shape5),
        (1,),
        device=device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (num_groups, 3, hidden),
        (3 * hidden, hidden, 1),
        device=device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        add_3,
        view_3,
        full,
        partials,
        ROWS=rows,
        HIDDEN=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_H=BLOCK_H,
        EPS_=EPS,
        INV_HIDDEN_=INV_HIDDEN,
        ROW_BACKWARD_SCALE_=ROW_BACKWARD_SCALE,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        num_warps=row_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partials,
        view_1,
        view_2,
        convert_element_type_4,
        NUM_GROUPS=num_groups,
        HIDDEN=hidden,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=final_warps,
    )

    permute = torch.as_strided(view_3, (hidden, rows), (1, hidden))
    return view_1, view_2, full, add_3, view_3, permute, convert_element_type_4
