"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Bart/Longformer layer-norm-backward/dropout tail by row-tiling the f32 producer, sharing the two hidden-dimension row reductions across the returned f32 gradient tensor, the bf16 mask-scaled view/permute side outputs, and the bf16-rounded channel sum, whereas Inductor schedules the row reductions, f32 side-output store, bf16 conversion/mask multiply, layout aliases, and final column reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalar reductions live while writing observable side outputs and coordinating compatible column partials; the fix is COOPERATIVE_SPLIT_K: add a dependent row-tiled producer/finalizer plan that keeps the row scalars live, emits the returned aliases with midpoint-safe bf16 store behavior, and finalizes the channel reduction from shared partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROW_FACTOR = 1024.0
DROP_SCALE = 1.1111111111111112


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
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    keep_ptr,
    grad_out_ptr,
    drop_out_ptr,
    partial_drop_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    drop_scale = tl.full((BLOCK_R, BLOCK_C), DROP_SCALE_, tl.float32).to(
        tl.bfloat16
    )
    row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32)

    acc_drop = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for row_offset in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + row_offset + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * HIDDEN + cols[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)

        weighted = _mul_rn(x, weight[None, :])
        weighted_rhs = _mul_rn(weighted, rhs)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(row_scale[:, None], centered)

        keep_bf16 = keep.to(tl.bfloat16)
        keep_scale_bf16 = (keep_bf16 * drop_scale).to(tl.bfloat16)
        exact_drop_bf16 = (
            grad.to(tl.bfloat16) * keep_scale_bf16
        ).to(tl.bfloat16)
        store_drop_bf16 = _mul_rn(
            grad,
            _mul_rn(keep, DROP_SCALE_ + 0.0),
        ).to(tl.bfloat16)
        grad_low_bits = grad.to(tl.uint32, bitcast=True) & 0xFFFF
        near_bf16_midpoint = (grad_low_bits >= 0x7FFD) & (grad_low_bits <= 0x8001)
        # Inductor fuses the visible store around bf16 midpoints; the final
        # column sum still follows the explicit bf16-boundary graph value.
        visible_drop_bf16 = tl.where(
            near_bf16_midpoint,
            store_drop_bf16.to(tl.float32),
            exact_drop_bf16.to(tl.float32),
        ).to(tl.bfloat16)

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(drop_out_ptr + offsets, visible_drop_bf16, mask=mask)
        acc_drop += tl.sum(
            tl.where(mask, exact_drop_bf16.to(tl.float32), 0.0), axis=0
        )

    partial_offsets = group * HIDDEN + cols
    tl.store(partial_drop_ptr + partial_offsets, acc_drop, mask=col_mask)


@triton.jit
def _finalize_drop_sum_kernel(
    partial_drop_ptr,
    out_drop_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * HIDDEN + cols[None, :]

    partial = tl.load(partial_drop_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    rounded = tl.sum(partial, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_drop_sum_ptr + cols, rounded, mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# c040a99e: (T([8,1024,1024], f32), T([1024], f32), T([8,1024,1024], f32), T([8,1024,1], f32), T([8,1024,1024], b8), ...)
@oracle_impl(
    hardware="B200",
    point="c040a99e",
    ROWS_PER_GROUP=8,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    num_warps=8,
    final_warps=8,
)
# f006a98f: (T([8,1024,768], f32), T([768], f32), T([8,1024,768], f32), T([8,1024,1], f32), T([8,1024,768], b8), ...)
@oracle_impl(
    hardware="B200",
    point="f006a98f",
    ROWS_PER_GROUP=8,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    num_warps=8,
    final_warps=8,
)
# befb65f3: (T([16,1024,768], f32), T([768], f32), T([16,1024,768], f32), T([16,1024,1], f32), T([16,1024,768], b8), ...)
@oracle_impl(
    hardware="B200",
    point="befb65f3",
    ROWS_PER_GROUP=16,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    num_warps=8,
    final_warps=8,
)
# bb79344b: (T([64,256,1024], f32), T([1024], f32), T([64,256,1024], f32), T([64,256,1], f32), T([64,256,1024], b8), ...)
@oracle_impl(
    hardware="B200",
    point="bb79344b",
    ROWS_PER_GROUP=16,
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs
    rows, hidden = _shape_tuple(shape0)

    grad_out = torch.empty_strided(
        tuple(int(dim) for dim in arg0_1.shape),
        tuple(int(stride) for stride in arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    drop_out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out_drop_sum = torch.empty_strided(
        _shape_tuple(shape1),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partial_drop = torch.empty_strided(
        (num_groups, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        grad_out,
        drop_out,
        partial_drop,
        ROWS=rows,
        HIDDEN=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        DROP_SCALE_=DROP_SCALE,
        num_warps=num_warps,
    )

    block_groups = 1 << (num_groups - 1).bit_length()
    _finalize_drop_sum_kernel[(triton.cdiv(hidden, FINAL_BLOCK_C),)](
        partial_drop,
        out_drop_sum,
        NUM_GROUPS=num_groups,
        HIDDEN=hidden,
        GROUP_BLOCK=block_groups,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    return grad_out, drop_out, drop_out.permute(1, 0), out_drop_sum
