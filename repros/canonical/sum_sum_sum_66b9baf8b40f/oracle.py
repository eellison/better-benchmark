"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full GoogleFnet fp32 layer-norm-backward return tuple by row-tiling the fixed `[16384, 768]` producer, sharing each row's hidden-dimension reductions across the returned input-gradient tensor, the zero-initialized `[32,512,768,2]` full tensor, the lane-0 `select_scatter` side output, and both `[768]` column reductions, whereas Inductor schedules the row reductions, gradient materialization, zero-fill/select_scatter materialization, and sibling `sum([0, 1])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reduction scalars, materialized scatter side outputs, and compatible column accumulators in one coordinated producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, finalize their partials once, and fuse the dependent zero-fill/select_scatter stores with the row producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 768
LANES = 2
ROW_FACTOR = 768.0


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
    mm_ptr,
    residual_ptr,
    gamma_ptr,
    xhat_ptr,
    scale_ptr,
    grad_ptr,
    full_ptr,
    scatter_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    LANES_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
):
    group = tl.program_id(0)
    rows = group * ROWS_PER_GROUP + tl.arange(0, ROWS_PER_GROUP)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS_
    col_mask = cols < HIDDEN_
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN_ + cols[None, :]

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    add = _add_rn(mm, residual)
    weighted = _mul_rn(add, gamma[None, :])
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, xhat), 0.0), axis=1)
    sub = _sub_rn(_mul_rn(weighted, ROW_FACTOR_), row_sum[:, None])
    sub_1 = _sub_rn(sub, _mul_rn(xhat, row_dot[:, None]))
    grad = _mul_rn(scale[:, None], sub_1)

    tl.store(grad_ptr + offsets, grad, mask=mask)

    side_offsets = offsets * LANES_
    zeros = tl.zeros((ROWS_PER_GROUP, BLOCK_H), dtype=tl.float32)
    tl.store(full_ptr + side_offsets, zeros, mask=mask)
    tl.store(full_ptr + side_offsets + 1, zeros, mask=mask)
    tl.store(scatter_ptr + side_offsets, grad, mask=mask)
    tl.store(scatter_ptr + side_offsets + 1, zeros, mask=mask)

    partial_sum_add_xhat = tl.sum(tl.where(mask, _mul_rn(add, xhat), 0.0), axis=0)
    partial_sum_add = tl.sum(tl.where(mask, add, 0.0), axis=0)
    partial_offsets = group * 2 * HIDDEN_ + cols
    tl.store(partials_ptr + partial_offsets, partial_sum_add_xhat, mask=col_mask)
    tl.store(partials_ptr + partial_offsets + HIDDEN_, partial_sum_add, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_sum_add_xhat_ptr,
    out_sum_add_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 2 * HIDDEN_ + cols[None, :]

    sum_add_xhat = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_add = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_sum_add_xhat_ptr + cols, tl.sum(sum_add_xhat, axis=0), mask=col_mask)
    tl.store(out_sum_add_ptr + cols, tl.sum(sum_add, axis=0), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="5297937a",
    ROWS_PER_GROUP=8,
    BLOCK_H=1024,
    FINAL_BLOCK_H=16,
    row_warps=8,
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
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        shape0,
        shape1,
    ) = inputs

    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)

    grad = torch.empty_strided(
        _shape_tuple(shape0),
        (int(shape0[1]) * hidden, hidden, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_add_xhat = torch.empty_strided((hidden,), (1,), device=device, dtype=torch.float32)
    sum_add = torch.empty_strided((hidden,), (1,), device=device, dtype=torch.float32)
    full = torch.empty_strided(
        _shape_tuple(shape1),
        (int(shape1[1]) * hidden * LANES, hidden * LANES, LANES, 1),
        device=device,
        dtype=torch.float32,
    )
    scatter = torch.empty_strided(
        _shape_tuple(shape1),
        (int(shape1[1]) * hidden * LANES, hidden * LANES, LANES, 1),
        device=device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (num_groups, 2, hidden),
        (2 * hidden, hidden, 1),
        device=device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        grad,
        full,
        scatter,
        partials,
        ROWS_=rows,
        HIDDEN_=hidden,
        LANES_=LANES,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_H=BLOCK_H,
        ROW_FACTOR_=ROW_FACTOR,
        num_warps=row_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partials,
        sum_add_xhat,
        sum_add,
        NUM_GROUPS=num_groups,
        HIDDEN_=hidden,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=final_warps,
    )

    return grad, sum_add_xhat, sum_add, full, scatter
