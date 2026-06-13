"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 exact-erf GELU/layer-norm-backward return tuple by row-tiling the `[rows, hidden]` producer, preserving the captured literal 1536 layer-norm factor/divisor, the bf16 rounding after GELU, after the layer-norm-backward term, and after the final GELU-gradient product, returning the contiguous bf16 gradient buffer plus its transpose alias, and cooperatively accumulating all three returned hidden-column reductions from the same row tiles, whereas Inductor lowers the GELU pointwise chain, row-local reductions, full-tensor stores, transpose view, and sibling column reductions as separate generic scheduled regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local hidden reductions, exact libdevice erf/exp epilogues, dtype-rounding boundaries, a dependent full-tensor side output, and multiple compatible column accumulators in one producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible transformer layer-norm-backward column reductions across row tiles, fuse the exact-GELU side-output producer, preserve explicit bf16 casts and captured constants, and finalize sibling column partials together."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
    x_ptr,
    gamma_ptr,
    gelu_x_ptr,
    center_ptr,
    scale_ptr,
    grad_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    ln_factor = tl.full((BLOCK_R, BLOCK_H), 1536.0, tl.float32)

    acc_x_norm = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_grad = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        offsets = rows[:, None] * HIDDEN + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu_x = tl.load(gelu_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        center = tl.load(center_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        erf_plus_one = _add_rn(libdevice.erf(_mul_rn(gelu_x, 0.7071067811865476)), 1.0)
        gelu_half = _mul_rn(gelu_x, 0.5)
        gelu_bf16 = _mul_rn(gelu_half, erf_plus_one).to(tl.bfloat16)
        norm = _mul_rn(_sub_rn(gelu_bf16.to(tl.float32), center[:, None]), scale[:, None])

        weighted = _mul_rn(x, gamma[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, norm), 0.0), axis=1)

        ln_num = _sub_rn(_mul_rn(weighted, ln_factor), row_sum[:, None])
        ln_num = _sub_rn(ln_num, _mul_rn(norm, row_dot[:, None]))
        ln_grad_bf16 = _mul_rn(scale[:, None] / 1536.0, ln_num).to(tl.bfloat16)

        exp_arg = _mul_rn(_mul_rn(gelu_x, gelu_x), -0.5)
        pdf = _mul_rn(libdevice.exp(exp_arg), 0.3989422804014327)
        gelu_grad = _add_rn(_mul_rn(erf_plus_one, 0.5), _mul_rn(gelu_x, pdf))
        grad_bf16 = _mul_rn(ln_grad_bf16.to(tl.float32), gelu_grad).to(tl.bfloat16)
        tl.store(grad_out_ptr + offsets, grad_bf16, mask=mask)

        acc_x_norm += tl.sum(tl.where(mask, _mul_rn(x, norm), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_grad += tl.sum(tl.where(mask, grad_bf16.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * HIDDEN + cols
    tl.store(partials_ptr + partial_base, acc_x_norm, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN, acc_grad, mask=col_mask)


@triton.jit
def _row_partials_h1536_kernel(
    x_ptr,
    gamma_ptr,
    gelu_x_ptr,
    center_ptr,
    scale_ptr,
    grad_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    group = tl.program_id(0)
    cols0 = tl.arange(0, 1024)
    cols1 = tl.arange(0, 512) + 1024
    gamma0 = tl.load(gamma_ptr + cols0).to(tl.float32)
    gamma1 = tl.load(gamma_ptr + cols1).to(tl.float32)
    ln_factor0 = tl.full((BLOCK_R, 1024), 1536.0, tl.float32)
    ln_factor1 = tl.full((BLOCK_R, 512), 1536.0, tl.float32)

    acc_x_norm0 = tl.zeros((1024,), dtype=tl.float32)
    acc_x0 = tl.zeros((1024,), dtype=tl.float32)
    acc_grad0 = tl.zeros((1024,), dtype=tl.float32)
    acc_x_norm1 = tl.zeros((512,), dtype=tl.float32)
    acc_x1 = tl.zeros((512,), dtype=tl.float32)
    acc_grad1 = tl.zeros((512,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        offsets0 = rows[:, None] * 1536 + cols0[None, :]
        offsets1 = rows[:, None] * 1536 + cols1[None, :]
        mask0 = row_mask[:, None]
        mask1 = row_mask[:, None]

        x0 = tl.load(x_ptr + offsets0, mask=mask0, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
        gelu_x0 = tl.load(gelu_x_ptr + offsets0, mask=mask0, other=0.0).to(tl.float32)
        gelu_x1 = tl.load(gelu_x_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
        center = tl.load(center_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        erf_plus_one0 = _add_rn(libdevice.erf(_mul_rn(gelu_x0, 0.7071067811865476)), 1.0)
        erf_plus_one1 = _add_rn(libdevice.erf(_mul_rn(gelu_x1, 0.7071067811865476)), 1.0)
        gelu_bf16_0 = _mul_rn(_mul_rn(gelu_x0, 0.5), erf_plus_one0).to(tl.bfloat16)
        gelu_bf16_1 = _mul_rn(_mul_rn(gelu_x1, 0.5), erf_plus_one1).to(tl.bfloat16)
        norm0 = _mul_rn(_sub_rn(gelu_bf16_0.to(tl.float32), center[:, None]), scale[:, None])
        norm1 = _mul_rn(_sub_rn(gelu_bf16_1.to(tl.float32), center[:, None]), scale[:, None])

        weighted0 = _mul_rn(x0, gamma0[None, :])
        weighted1 = _mul_rn(x1, gamma1[None, :])
        row_sum = (
            tl.sum(tl.where(mask0, weighted0, 0.0), axis=1)
            + tl.sum(tl.where(mask1, weighted1, 0.0), axis=1)
        )
        row_dot = (
            tl.sum(tl.where(mask0, _mul_rn(weighted0, norm0), 0.0), axis=1)
            + tl.sum(tl.where(mask1, _mul_rn(weighted1, norm1), 0.0), axis=1)
        )

        ln_num0 = _sub_rn(_mul_rn(weighted0, ln_factor0), row_sum[:, None])
        ln_num0 = _sub_rn(ln_num0, _mul_rn(norm0, row_dot[:, None]))
        ln_num1 = _sub_rn(_mul_rn(weighted1, ln_factor1), row_sum[:, None])
        ln_num1 = _sub_rn(ln_num1, _mul_rn(norm1, row_dot[:, None]))
        ln_grad_bf16_0 = _mul_rn(scale[:, None] / 1536.0, ln_num0).to(tl.bfloat16)
        ln_grad_bf16_1 = _mul_rn(scale[:, None] / 1536.0, ln_num1).to(tl.bfloat16)

        pdf0 = _mul_rn(
            libdevice.exp(_mul_rn(_mul_rn(gelu_x0, gelu_x0), -0.5)),
            0.3989422804014327,
        )
        pdf1 = _mul_rn(
            libdevice.exp(_mul_rn(_mul_rn(gelu_x1, gelu_x1), -0.5)),
            0.3989422804014327,
        )
        gelu_grad0 = _add_rn(_mul_rn(erf_plus_one0, 0.5), _mul_rn(gelu_x0, pdf0))
        gelu_grad1 = _add_rn(_mul_rn(erf_plus_one1, 0.5), _mul_rn(gelu_x1, pdf1))
        grad_bf16_0 = _mul_rn(ln_grad_bf16_0.to(tl.float32), gelu_grad0).to(tl.bfloat16)
        grad_bf16_1 = _mul_rn(ln_grad_bf16_1.to(tl.float32), gelu_grad1).to(tl.bfloat16)
        tl.store(grad_out_ptr + offsets0, grad_bf16_0, mask=mask0)
        tl.store(grad_out_ptr + offsets1, grad_bf16_1, mask=mask1)

        acc_x_norm0 += tl.sum(tl.where(mask0, _mul_rn(x0, norm0), 0.0), axis=0)
        acc_x0 += tl.sum(tl.where(mask0, x0, 0.0), axis=0)
        acc_grad0 += tl.sum(tl.where(mask0, grad_bf16_0.to(tl.float32), 0.0), axis=0)
        acc_x_norm1 += tl.sum(tl.where(mask1, _mul_rn(x1, norm1), 0.0), axis=0)
        acc_x1 += tl.sum(tl.where(mask1, x1, 0.0), axis=0)
        acc_grad1 += tl.sum(tl.where(mask1, grad_bf16_1.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * 1536
    tl.store(partials_ptr + partial_base + cols0, acc_x_norm0)
    tl.store(partials_ptr + partial_base + cols1, acc_x_norm1)
    tl.store(partials_ptr + partial_base + 1536 + cols0, acc_x0)
    tl.store(partials_ptr + partial_base + 1536 + cols1, acc_x1)
    tl.store(partials_ptr + partial_base + 3072 + cols0, acc_grad0)
    tl.store(partials_ptr + partial_base + 3072 + cols1, acc_grad1)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    sum_x_norm_ptr,
    sum_x_ptr,
    sum_grad_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN
    group_offsets = tl.arange(0, BLOCK_GROUPS)

    acc_x_norm = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_grad = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for group_base in range(0, NUM_GROUPS, BLOCK_GROUPS):
        groups = group_base + group_offsets
        mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
        offsets = groups[:, None] * 3 * HIDDEN + cols[None, :]
        acc_x_norm += tl.sum(
            tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partials_ptr + offsets + HIDDEN, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_grad += tl.sum(
            tl.load(partials_ptr + offsets + 2 * HIDDEN, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    tl.store(sum_x_norm_ptr + cols, acc_x_norm, mask=col_mask)
    tl.store(sum_x_ptr + cols, acc_x, mask=col_mask)
    tl.store(sum_grad_ptr + cols, acc_grad.to(tl.bfloat16).to(tl.float32), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="49fb3d4b", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=2048, FINAL_BLOCK_H=8, FINAL_BLOCK_GROUPS=512, num_warps=8)
@oracle_impl(hardware="B200", point="e35c96d4", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, FINAL_BLOCK_GROUPS=1024, num_warps=8)
@oracle_impl(hardware="B200", point="02c9dc6d", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, FINAL_BLOCK_GROUPS=1024, num_warps=8)
@oracle_impl(hardware="B200", point="37c6392e", ROWS_PER_GROUP=32, BLOCK_R=1, BLOCK_H=128, FINAL_BLOCK_H=16, FINAL_BLOCK_GROUPS=1024, num_warps=4)
@oracle_impl(hardware="B200", point="248b11c9", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, FINAL_BLOCK_GROUPS=512, num_warps=8)
@oracle_impl(hardware="B200", point="138859e5", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8, FINAL_BLOCK_GROUPS=512, num_warps=8)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
    FINAL_BLOCK_GROUPS: int,
    num_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        _shape_param_0,
        _shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs

    out_shape = _shape_tuple(shape_param_2)
    sum_shape = _shape_tuple(shape_param_3)
    rows = int(out_shape[0])
    hidden = int(out_shape[1])

    grad_out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_x_norm = torch.empty_strided(sum_shape, (1,), device=arg0_1.device, dtype=torch.float32)
    sum_x = torch.empty_strided(sum_shape, (1,), device=arg0_1.device, dtype=torch.float32)
    sum_grad = torch.empty_strided(sum_shape, (1,), device=arg0_1.device, dtype=torch.float32)

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, hidden),
        (3 * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    if hidden == 1536:
        _row_partials_h1536_kernel[(num_groups,)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            grad_out,
            partials,
            ROWS=rows,
            ROWS_PER_GROUP=ROWS_PER_GROUP,
            BLOCK_R=BLOCK_R,
            num_warps=num_warps,
        )
    else:
        _row_partials_kernel[(num_groups,)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            grad_out,
            partials,
            ROWS=rows,
            HIDDEN=hidden,
            ROWS_PER_GROUP=ROWS_PER_GROUP,
            BLOCK_R=BLOCK_R,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
        )
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partials,
        sum_x_norm,
        sum_x,
        sum_grad,
        NUM_GROUPS=num_groups,
        HIDDEN=hidden,
        BLOCK_GROUPS=FINAL_BLOCK_GROUPS,
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=8,
    )

    return sum_x_norm, sum_x, grad_out, grad_out.permute(1, 0), sum_grad
