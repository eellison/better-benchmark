"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GoogleFnet fp32 tanh-GELU/layer-norm-backward scope by row-tiling the fixed `[16384, 768]` producer, sharing each row's hidden-dimension reductions across the returned fp32 GELU-gradient tensor, its metadata-only transpose alias, and all three `[768]` column reductions, whereas Inductor schedules the tanh-GELU producer, row-local layer-norm reductions, dense side-output store, alias return, and sibling column reductions as separate generic pointwise and reduction regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that keeps row-local layer-norm scalars live through a tanh-GELU backward epilogue while also emitting compatible column partials and view aliases from one producer; the fix is COOPERATIVE_SPLIT_K: add a guarded GoogleFnet row-tiled layer-norm/GELU-backward schedule that writes the observable dense output once, returns alias-only layout views, and finalizes sibling column reductions directly from cooperative partials."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 768
ROW_FACTOR = 768.0
INV_HIDDEN = 1.0 / 768.0
SQRT_2_OVER_PI = 0.7978845608028654
GELU_TANH_CUBE = 0.044715


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
def _row_partials_kernel(
    ln_input_ptr,
    weight_ptr,
    gelu_input_ptr,
    center_ptr,
    row_scale_ptr,
    grad_out_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    INV_HIDDEN_: tl.constexpr,
    SQRT_2_OVER_PI_: tl.constexpr,
    GELU_TANH_CUBE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_times_normed = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_grad = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(ln_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu_x = tl.load(gelu_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        center = tl.load(center_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        weighted = _mul_rn(x, weight[None, :])
        weighted_times_hidden = _mul_rn(weighted, ROW_FACTOR_)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)

        gelu_half = _mul_rn(gelu_x, 0.5)
        gelu_x2 = _mul_rn(gelu_x, gelu_x)
        gelu_x3 = _mul_rn(gelu_x2, gelu_x)
        scaled_cubic = _mul_rn(gelu_x3, GELU_TANH_CUBE_)
        gelu_inner = _add_rn(gelu_x, scaled_cubic)
        tanh_arg = _mul_rn(gelu_inner, SQRT_2_OVER_PI_)
        tanh_val = libdevice.tanh(tanh_arg)
        tanh_plus_one = _add_rn(tanh_val, 1.0)
        gelu = _mul_rn(gelu_half, tanh_plus_one)
        normed = _mul_rn(_sub_rn(gelu, center[:, None]), row_scale[:, None])

        weighted_normed = _mul_rn(weighted, normed)
        row_dot = tl.sum(tl.where(mask, weighted_normed, 0.0), axis=1)
        normed_row_dot = _mul_rn(normed, row_dot[:, None])
        centered = _sub_rn(_sub_rn(weighted_times_hidden, row_sum[:, None]), normed_row_dot)
        ln_grad = _mul_rn(_mul_rn(row_scale[:, None], INV_HIDDEN_), centered)

        mul11 = _mul_rn(ln_grad, gelu_half)
        mul12 = _mul_rn(ln_grad, tanh_plus_one)
        tanh_sq = _mul_rn(tanh_val, tanh_val)
        sech2 = _sub_rn(1.0, tanh_sq)
        mul14 = _mul_rn(mul11, sech2)
        mul15 = _mul_rn(mul14, SQRT_2_OVER_PI_)
        mul16 = _mul_rn(mul15, GELU_TANH_CUBE_)
        mul17 = _mul_rn(gelu_x2, 3.0)
        mul18 = _mul_rn(mul16, mul17)
        add2 = _add_rn(mul15, mul18)
        mul19 = _mul_rn(mul12, 0.5)
        grad = _add_rn(add2, mul19)

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        acc_x_times_normed += tl.sum(tl.where(mask, _mul_rn(x, normed), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_grad += tl.sum(tl.where(mask, grad, 0.0), axis=0)

    partial_base = group * 3 * HIDDEN_ + cols
    tl.store(partials_ptr + partial_base, acc_x_times_normed, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN_, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN_, acc_grad, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_times_normed_ptr,
    out_x_ptr,
    out_grad_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * HIDDEN_ + cols[None, :]

    x_times_normed = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(partials_ptr + offsets + 2 * HIDDEN_, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_times_normed_ptr + cols, tl.sum(x_times_normed, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_grad_sum_ptr + cols, tl.sum(grad, axis=0), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="930536e2",
    ROWS_PER_GROUP=8,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=4,
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
    row_warps: int,
    final_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    del _shape_param_0, _shape_param_1

    device = arg0.device
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])
    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)

    out_x_times_normed = torch.empty_strided(
        _shape_tuple(_shape_param_3),
        (1,),
        device=device,
        dtype=torch.float32,
    )
    out_x = torch.empty_strided(
        _shape_tuple(_shape_param_3),
        (1,),
        device=device,
        dtype=torch.float32,
    )
    grad_out = torch.empty_strided(
        _shape_tuple(_shape_param_2),
        (hidden, 1),
        device=device,
        dtype=torch.float32,
    )
    out_grad_sum = torch.empty_strided(
        _shape_tuple(_shape_param_3),
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
        grad_out,
        partials,
        ROWS_=rows,
        HIDDEN_=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        INV_HIDDEN_=INV_HIDDEN,
        SQRT_2_OVER_PI_=SQRT_2_OVER_PI,
        GELU_TANH_CUBE_=GELU_TANH_CUBE,
        num_warps=row_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_C),)](
        partials,
        out_x_times_normed,
        out_x,
        out_grad_sum,
        NUM_GROUPS=num_groups,
        HIDDEN_=hidden,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    return out_x_times_normed, out_x, grad_out, grad_out.permute(1, 0), out_grad_sum
