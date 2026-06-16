"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ALBERT bf16 tanh-GELU/layer-norm-backward return tuple by row-tiling the fixed `[4096,128]` producer, preserving the bf16 half-input and GELU-gradient cast/add boundaries, storing the returned bf16 gradient tensor and its `[128,4096]` permute alias, and cooperatively accumulating the two f32 source column reductions plus the bf16-rounded output column sum from the same row producer, whereas Inductor currently schedules the tanh-GELU algebra, row-local layer-norm reductions, full tensor materialization, alias return, and sibling column reductions as separate generic regions over materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions live through a dtype-boundary-sensitive GELU-backward epilogue while also emitting compatible column partials and view aliases; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible transformer layer-norm-backward column reductions across row tiles, fuse the tanh-GELU side-output producer, preserve explicit bf16 casts, and finalize sibling column partials together."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


ROWS = 4096
HIDDEN = 128
ROW_FACTOR = 128.0
INV_HIDDEN = 0.0078125
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
def _row_group_kernel(
    ln_input_ptr,
    weight_ptr,
    gelu_input_ptr,
    center_ptr,
    row_scale_ptr,
    grad_out_ptr,
    partials_ptr,
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
    col_mask = cols < 128
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_normed = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_grad = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < 4096
        offsets = rows[:, None] * 128 + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(ln_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu_bf16 = tl.load(gelu_input_ptr + offsets, mask=mask, other=0.0)
        gelu_x = gelu_bf16.to(tl.float32)
        center = tl.load(center_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        weighted = _mul_rn(x, weight[None, :])
        weighted_times_hidden = _mul_rn(weighted, ROW_FACTOR_)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)

        half_x = _mul_rn(gelu_x, 0.5).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        gelu_x2 = _mul_rn(gelu_x, gelu_x)
        gelu_x3 = _mul_rn(gelu_x2, gelu_x)
        scaled_cubic = _mul_rn(gelu_x3, GELU_TANH_CUBE_)
        tanh_arg = _mul_rn(_add_rn(gelu_x, scaled_cubic), SQRT_2_OVER_PI_)
        tanh_val = libdevice.tanh(tanh_arg)
        tanh_plus_one = _add_rn(tanh_val, 1.0)
        gelu_forward = _mul_rn(half_x, tanh_plus_one)
        normed = _mul_rn(_sub_rn(gelu_forward, center[:, None]), row_scale[:, None])

        weighted_normed = _mul_rn(weighted, normed)
        row_dot = tl.sum(tl.where(mask, weighted_normed, 0.0), axis=1)
        sub1 = _sub_rn(weighted_times_hidden, row_sum[:, None])
        sub2 = _sub_rn(sub1, _mul_rn(normed, row_dot[:, None]))
        ln_grad = _mul_rn(_mul_rn(row_scale[:, None], INV_HIDDEN_), sub2)

        mul11 = _mul_rn(ln_grad, half_x)
        mul12 = _mul_rn(ln_grad, tanh_plus_one)
        mul12_bf16 = mul12.to(tl.bfloat16, fp_downcast_rounding="rtne")
        tanh_sq = _mul_rn(tanh_val, tanh_val)
        sech2 = _sub_rn(1.0, tanh_sq)
        mul15 = _mul_rn(_mul_rn(_mul_rn(mul11, sech2), SQRT_2_OVER_PI_), 1.0)
        tail0 = mul15.to(tl.bfloat16, fp_downcast_rounding="rtne")
        mul16 = _mul_rn(mul15, GELU_TANH_CUBE_)
        mul17 = _mul_rn(gelu_x2, 3.0)
        tail1 = _mul_rn(mul16, mul17).to(tl.bfloat16, fp_downcast_rounding="rtne")
        tail = _add_rn(tail0.to(tl.float32), tail1.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        right = _mul_rn(mul12_bf16.to(tl.float32), 0.5).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
        grad = _add_rn(tail.to(tl.float32), right.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        grad_f32 = grad.to(tl.float32)
        acc_x_normed += tl.sum(tl.where(mask, _mul_rn(x, normed), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_grad += tl.sum(tl.where(mask, grad_f32, 0.0), axis=0)

    partial_base = group * 3 * 128 + cols
    tl.store(partials_ptr + partial_base, acc_x_normed, mask=col_mask)
    tl.store(partials_ptr + partial_base + 128, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 256, acc_grad, mask=col_mask)


@triton.jit
def _finalize_kernel(
    partials_ptr,
    out_x_normed_ptr,
    out_x_ptr,
    out_grad_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < 128
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * 128 + cols[None, :]

    x_normed = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + 128, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(partials_ptr + offsets + 256, mask=mask, other=0.0).to(tl.float32)

    grad_sum = tl.sum(grad, axis=0)
    grad_sum_rounded = grad_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_x_normed_ptr + cols, tl.sum(x_normed, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_grad_sum_ptr + cols, grad_sum_rounded, mask=col_mask)


# (T([4096,128], bf16), T([128], f32), T([4096,128], bf16), T([8,512,1], f32), T([8,512,1], f32), S([8,512,128]), S([8,512,128]), S([4096,128]), S([128]))
@oracle_impl(hardware="B200", point="bcddd3bd", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=128, FINAL_BLOCK_C=16, row_warps=4, final_warps=8)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, _shape2, _shape3 = inputs
    del _shape0, _shape1, _shape2, _shape3

    device = arg0_1.device
    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)

    out_x_normed = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    grad_out = torch.empty_strided((ROWS, HIDDEN), (HIDDEN, 1), device=device, dtype=torch.bfloat16)
    out_grad_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    partials = torch.empty((num_groups, 3, HIDDEN), device=device, dtype=torch.float32)

    _row_group_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        grad_out,
        partials,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        INV_HIDDEN_=INV_HIDDEN,
        SQRT_2_OVER_PI_=SQRT_2_OVER_PI,
        GELU_TANH_CUBE_=GELU_TANH_CUBE,
        num_warps=row_warps,
        num_stages=3,
    )

    _finalize_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_C),)](
        partials,
        out_x_normed,
        out_x,
        out_grad_sum,
        NUM_GROUPS=num_groups,
        GROUP_BLOCK=triton.next_power_of_2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )

    return out_x_normed, out_x, grad_out, grad_out.permute(1, 0), out_grad_sum
