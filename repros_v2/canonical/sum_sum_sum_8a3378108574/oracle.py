"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle streams the three bf16 matmul outputs once, widens them to fp32, computes the row-local layer-norm-backward reductions, writes the returned fp32 update plus its bf16 view/transpose alias, and cooperatively finalizes all three returned hidden-column reductions from shared row-block partials, whereas Inductor schedules the sibling column reductions, row reductions, dense update, bf16 cast/view fanout, transpose alias, and final bf16-rounded column reduction as separate generic regions; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local summaries, dtype round-trip boundaries, dependent full-tensor side outputs, and compatible column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layernorm-backward column reductions across token-row tiles, fuse the dense and bf16 side-output stores, and finalize all sibling column partials together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


HIDDEN = 2048
ROW_SPLIT = 4
XBLOCK = 1
FINAL_BLOCK_H = 16


@triton.jit
def _row_store_and_partial_reduce_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    gamma_ptr,
    rhs_ptr,
    row_scale_ptr,
    residual_ptr,
    add_out_ptr,
    bf16_out_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_bf16_out_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    ROW_SPLIT_: tl.constexpr,
    XBLOCK_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_bf16_out = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
        rows = pid * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
        row_mask = rows < ROWS_
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]

        x = (
            tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        x = x + tl.load(c_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )

        weighted = x * gamma[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        update = row_scale[:, None] * (
            weighted * HIDDEN_ - row_sum[:, None] - rhs * row_dot[:, None]
        )
        out = residual + update
        out_bf16 = out.to(tl.bfloat16)

        tl.store(add_out_ptr + offsets, out, mask=mask)
        tl.store(bf16_out_ptr + offsets, out_bf16, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_bf16_out += tl.sum(
            tl.where(mask, out_bf16.to(tl.float32), 0.0), axis=0
        )

    partial_offsets = pid * HIDDEN_ + cols
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)
    tl.store(partial_bf16_out_ptr + partial_offsets, acc_bf16_out, mask=col_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_bf16_out_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_bf16_sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    col_mask = cols < HIDDEN_
    mask = (row_blocks[:, None] < NUM_ROW_BLOCKS) & col_mask[None, :]
    offsets = row_blocks[:, None] * HIDDEN_ + cols[None, :]

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bf16_out = tl.load(partial_bf16_out_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    bf16_sum = tl.sum(bf16_out, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_bf16_sum_ptr + cols, bf16_sum, mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="b9b3dbcc",
    BLOCK_H=HIDDEN,
    FINAL_BLOCK=FINAL_BLOCK_H,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_H, FINAL_BLOCK, num_warps):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs

    view_shape = tuple(int(dim) for dim in _shape_param_0)
    flat_shape = tuple(int(dim) for dim in _shape_param_3)
    vec_shape = tuple(int(dim) for dim in _shape_param_4)
    rows = int(flat_shape[0])
    hidden = int(flat_shape[1])
    num_row_blocks = triton.cdiv(rows, ROW_SPLIT)

    a = arg0_1.reshape(rows, hidden)
    b = arg1_1.reshape(rows, hidden)
    c = arg2_1.reshape(rows, hidden)
    rhs = arg4_1.reshape(rows, hidden)
    row_scale = arg5_1.reshape(rows)
    residual = arg6_1.reshape(rows, hidden)

    add_out = torch.empty_strided(
        view_shape,
        tuple(int(s) for s in arg6_1.stride()),
        device=arg6_1.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial_shape = (num_row_blocks, hidden)
    partial_x_rhs = torch.empty(
        partial_shape, device=arg0_1.device, dtype=torch.float32
    )
    partial_x = torch.empty(partial_shape, device=arg0_1.device, dtype=torch.float32)
    partial_bf16_out = torch.empty(
        partial_shape, device=arg0_1.device, dtype=torch.float32
    )

    _row_store_and_partial_reduce_kernel[(num_row_blocks,)](
        a,
        b,
        c,
        arg3_1,
        rhs,
        row_scale,
        residual,
        add_out,
        bf16_out,
        partial_x_rhs,
        partial_x,
        partial_bf16_out,
        ROWS_=rows,
        HIDDEN_=hidden,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )

    sum_x_rhs = torch.empty(vec_shape, device=arg0_1.device, dtype=torch.float32)
    sum_x = torch.empty(vec_shape, device=arg0_1.device, dtype=torch.float32)
    bf16_sum = torch.empty(vec_shape, device=arg0_1.device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK),)](
        partial_x_rhs,
        partial_x,
        partial_bf16_out,
        sum_x_rhs,
        sum_x,
        bf16_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        HIDDEN_=hidden,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_H=FINAL_BLOCK,
        num_warps=8,
    )

    return sum_x_rhs, sum_x, add_out, bf16_out, bf16_out.permute(1, 0), bf16_sum
