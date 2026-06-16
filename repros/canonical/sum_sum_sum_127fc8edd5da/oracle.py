"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DistilBERT bf16 LayerNorm-backward tail by rebuilding the bf16-to-fp32 residual add producer inside a row-tiled Triton kernel, reducing each hidden row for the dependent gradient formula, writing both the returned fp32 dense gradient and the bf16-rounded flat/transpose side output, and cooperatively accumulating the two pre-output `[768]` column reductions plus the post-bf16-cast `[768]` column reduction from the same tiles, whereas Inductor schedules the shared producer, row-local reductions, dense epilogue, dtype-conversion side output, transpose view, and sibling column reductions as separate generic regions over materialized or replayed intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps explicit bf16 cast boundaries, row-local scalar reductions, visible side-output stores, and multiple compatible column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible LayerNorm-backward column reductions across token tiles, fuse the shared producer plus side-output stores, and finalize all sibling column partials together."""

import torch
import triton
import triton.language as tl

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
def _row_partials_and_side_kernel(
    arg0_ptr,
    arg1_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    dense_f32_ptr,
    dense_bf16_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_bf16_dense_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    row_group: tl.constexpr,
    xblock: tl.constexpr,
    block_hidden: tl.constexpr,
):
    row_group_id = tl.program_id(0)
    hidden_offsets = tl.arange(0, block_hidden)
    hidden_mask = hidden_offsets < hidden
    weight = tl.load(weight_ptr + hidden_offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )
    row_factor = tl.full((xblock, block_hidden), 768.0, tl.float32)

    acc_x_rhs = tl.zeros((block_hidden,), dtype=tl.float32)
    acc_x = tl.zeros((block_hidden,), dtype=tl.float32)
    acc_bf16_dense = tl.zeros((block_hidden,), dtype=tl.float32)

    for group_offset in tl.range(0, row_group, xblock):
        row_offsets = row_group_id * row_group + group_offset + tl.arange(0, xblock)
        row_mask = row_offsets < rows
        mask = row_mask[:, None] & hidden_mask[None, :]
        offsets = row_offsets[:, None] * hidden + hidden_offsets[None, :]

        x = _add_rn(
            tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        )
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = _mul_rn(x, weight[None, :])
        weighted_rhs = _mul_rn(weighted, rhs)

        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)
        gate = tl.load(gate_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        row_sum = _add_rn(
            row_sum,
            tl.where(gate < 0.0, 2.86102294921875e-6, -7.62939453125e-6),
        )
        scaled = _mul_rn(weighted, row_factor)
        centered = _sub_rn(scaled, row_sum[:, None])
        projected = _mul_rn(rhs, row_dot[:, None])
        grad_inner = _sub_rn(centered, projected)
        dense_f32 = _mul_rn(gate[:, None], grad_inner)
        dense_bf16 = dense_f32.to(tl.bfloat16)

        tl.store(dense_f32_ptr + offsets, dense_f32, mask=mask)
        tl.store(dense_bf16_ptr + offsets, dense_bf16, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, _mul_rn(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_bf16_dense += tl.sum(
            tl.where(mask, dense_bf16.to(tl.float32), 0.0), axis=0
        )

    partial_offsets = row_group_id * hidden + hidden_offsets
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=hidden_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=hidden_mask)
    tl.store(
        partial_bf16_dense_ptr + partial_offsets, acc_bf16_dense, mask=hidden_mask
    )


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_bf16_dense_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_bf16_dense_ptr,
    num_row_groups: tl.constexpr,
    hidden: tl.constexpr,
    block_groups: tl.constexpr,
    block_hidden: tl.constexpr,
):
    hidden_offsets = tl.program_id(0) * block_hidden + tl.arange(0, block_hidden)
    hidden_mask = hidden_offsets < hidden
    group_offsets = tl.arange(0, block_groups)

    acc_x_rhs = tl.zeros((block_hidden,), dtype=tl.float32)
    acc_x = tl.zeros((block_hidden,), dtype=tl.float32)
    acc_bf16_dense = tl.zeros((block_hidden,), dtype=tl.float32)

    for group_start in range(0, num_row_groups, block_groups):
        groups = group_start + group_offsets
        mask = (groups[:, None] < num_row_groups) & hidden_mask[None, :]
        offsets = groups[:, None] * hidden + hidden_offsets[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_bf16_dense += tl.sum(
            tl.load(partial_bf16_dense_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + hidden_offsets, acc_x_rhs, mask=hidden_mask)
    tl.store(out_x_ptr + hidden_offsets, acc_x, mask=hidden_mask)
    tl.store(
        out_bf16_dense_ptr + hidden_offsets,
        acc_bf16_dense.to(tl.bfloat16).to(tl.float32),
        mask=hidden_mask,
    )


# 953b9872: (T([32768,768], bf16), T([256,128,768], f32), ...)
@oracle_impl(
    hardware="B200",
    point="953b9872",
    row_group=16,
    xblock=1,
    block_hidden=1024,
    final_block_groups=256,
    final_block_hidden=16,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    row_group: int,
    xblock: int,
    block_hidden: int,
    final_block_groups: int,
    final_block_hidden: int,
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
        shape2,
    ) = inputs

    full_shape = tuple(int(dim) for dim in shape0)
    flat_shape = tuple(int(dim) for dim in shape1)
    sum_shape = tuple(int(dim) for dim in shape2)
    rows = int(flat_shape[0])
    hidden = int(flat_shape[1])
    num_row_groups = triton.cdiv(rows, row_group)

    dense_f32 = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    dense_bf16 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    partial_x_rhs = torch.empty(
        (num_row_groups, hidden), device=arg1_1.device, dtype=torch.float32
    )
    partial_x = torch.empty_like(partial_x_rhs)
    partial_bf16_dense = torch.empty_like(partial_x_rhs)
    out_x_rhs = torch.empty_strided(
        sum_shape, (1,), device=arg1_1.device, dtype=torch.float32
    )
    out_x = torch.empty_strided(
        sum_shape, (1,), device=arg1_1.device, dtype=torch.float32
    )
    out_bf16_dense = torch.empty_strided(
        sum_shape, (1,), device=arg1_1.device, dtype=torch.float32
    )

    _row_partials_and_side_kernel[(num_row_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        dense_f32,
        dense_bf16,
        partial_x_rhs,
        partial_x,
        partial_bf16_dense,
        rows=rows,
        hidden=hidden,
        row_group=row_group,
        xblock=xblock,
        block_hidden=block_hidden,
        num_warps=row_warps,
    )
    _finalize_column_sums_kernel[(triton.cdiv(hidden, final_block_hidden),)](
        partial_x_rhs,
        partial_x,
        partial_bf16_dense,
        out_x_rhs,
        out_x,
        out_bf16_dense,
        num_row_groups=num_row_groups,
        hidden=hidden,
        block_groups=final_block_groups,
        block_hidden=final_block_hidden,
        num_warps=final_warps,
    )
    return (
        dense_f32,
        out_x_rhs,
        out_x,
        dense_bf16,
        dense_bf16.permute(1, 0),
        out_bf16_dense,
    )
