"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 BERT-family layer-norm-backward/dropout return tuple by reducing each token row once, writing both the fp32 gradient and bf16 dropout-masked flat/permute side outputs, and cooperatively accumulating the two pre-mask hidden reductions plus the rounded post-mask hidden reduction from the same row tiles, whereas Inductor schedules the row reductions, bf16 cast/mask epilogue, returned view/permute materialization, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, dtype-rounding side stores, aliasing view outputs, and multiple compatible column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split layer-norm-backward column reductions across token-row tiles, preserve bf16 cast boundaries while writing the side output, and finalize the sibling column partials together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


DROP_SCALE = 1.1111111111111112


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_group_store_and_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    keep_ptr,
    grad_ptr,
    masked_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    NORM_SIZE: tl.constexpr,
    DROP_SCALE: tl.constexpr,
    ROW_GROUP: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_group = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for group_offset in tl.range(0, ROW_GROUP, XBLOCK):
        m = row_group * ROW_GROUP + group_offset + tl.arange(0, XBLOCK)
        m_mask = m < M
        mask = m_mask[:, None] & c_mask[None, :]
        offsets = m[:, None] * C + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + m, mask=m_mask, other=0.0).to(
            tl.float32
        )

        weighted = _f32_mul(x, weight[None, :])
        weighted_rhs = _f32_mul(weighted, rhs)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)
        scaled = _f32_mul(weighted, NORM_SIZE + 0.0)
        sub = _f32_sub(scaled, row_sum[:, None])
        rhs_row_dot = _f32_mul(rhs, row_dot[:, None])
        sub_1 = _f32_sub(sub, rhs_row_dot)
        grad = _f32_mul(row_scale[:, None], sub_1)
        tl.store(grad_ptr + offsets, grad, mask=mask)

        keep = tl.load(keep_ptr + offsets, mask=mask, other=0)
        keep_bf16 = keep.to(tl.bfloat16)
        drop = (keep_bf16 * DROP_SCALE).to(tl.bfloat16)
        current_bf16 = (grad.to(tl.bfloat16) * drop).to(tl.bfloat16)
        keep_scale = _f32_mul(keep.to(tl.float32), DROP_SCALE + 0.0)
        midpoint_bf16 = _f32_mul(grad, keep_scale).to(tl.bfloat16)
        grad_bits = grad.to(tl.uint32, bitcast=True)
        grad_low_bits = grad_bits & 0xFFFF
        near_bf16_midpoint = (grad_low_bits >= 0x7FFF) & (grad_low_bits <= 0x8001)
        masked_bf16 = tl.where(
            near_bf16_midpoint,
            midpoint_bf16.to(tl.float32),
            current_bf16.to(tl.float32),
        ).to(tl.bfloat16)
        tl.store(masked_ptr + offsets, masked_bf16, mask=mask)

        x_rhs = _f32_mul(x, rhs)
        acc_x_rhs += tl.sum(tl.where(mask, x_rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_masked += tl.sum(tl.where(mask, masked_bf16.to(tl.float32), 0.0), axis=0)

    partial_offsets = row_group * C + c
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
    tl.store(partial_masked_ptr + partial_offsets, acc_masked, mask=c_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_masked_ptr,
    NUM_ROW_GROUPS: tl.constexpr,
    C: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C
    groups = tl.arange(0, BLOCK_GROUPS)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for group_start in range(0, NUM_ROW_GROUPS, BLOCK_GROUPS):
        group_ids = group_start + groups
        mask = (group_ids[:, None] < NUM_ROW_GROUPS) & c_mask[None, :]
        offsets = group_ids[:, None] * C + c[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_masked += tl.sum(
            tl.load(partial_masked_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
    tl.store(out_x_ptr + c, acc_x, mask=c_mask)
    tl.store(out_masked_ptr + c, acc_masked.to(tl.bfloat16).to(tl.float32), mask=c_mask)


@oracle_impl(
    hardware="B200",
    point="09b68f1c",
    ROW_GROUP=8,
    XBLOCK=1,
    BLOCK_C=2048,
    FINAL_BLOCK_C=16,
    FINAL_BLOCK_GROUPS=512,
    ROW_NUM_WARPS=8,
    FINAL_NUM_WARPS=8,
)
@oracle_impl(
    hardware="B200",
    point="3fb352aa",
    ROW_GROUP=16,
    XBLOCK=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    FINAL_BLOCK_GROUPS=1024,
    ROW_NUM_WARPS=8,
    FINAL_NUM_WARPS=8,
)
@oracle_impl(
    hardware="B200",
    point="dca7eb09",
    ROW_GROUP=16,
    XBLOCK=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    FINAL_BLOCK_GROUPS=1024,
    ROW_NUM_WARPS=8,
    FINAL_NUM_WARPS=8,
)
@oracle_impl(
    hardware="B200",
    point="8fb12abb",
    ROW_GROUP=32,
    XBLOCK=1,
    BLOCK_C=256,
    FINAL_BLOCK_C=16,
    FINAL_BLOCK_GROUPS=1024,
    ROW_NUM_WARPS=8,
    FINAL_NUM_WARPS=8,
)
@oracle_impl(
    hardware="B200",
    point="a1f869c6",
    ROW_GROUP=16,
    XBLOCK=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    FINAL_BLOCK_GROUPS=512,
    ROW_NUM_WARPS=8,
    FINAL_NUM_WARPS=8,
)
def oracle_forward(
    inputs,
    *,
    ROW_GROUP: int,
    XBLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    FINAL_BLOCK_GROUPS: int,
    ROW_NUM_WARPS: int,
    FINAL_NUM_WARPS: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    batch = int(arg2_1.shape[0])
    seq = int(arg2_1.shape[1])
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    num_row_groups = triton.cdiv(rows, ROW_GROUP)

    grad = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    masked = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (3, num_row_groups, hidden),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _row_group_store_and_reduce_kernel[(num_row_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        grad,
        masked,
        partials[0],
        partials[1],
        partials[2],
        M=rows,
        C=hidden,
        NORM_SIZE=1536,
        DROP_SCALE=DROP_SCALE,
        ROW_GROUP=ROW_GROUP,
        XBLOCK=XBLOCK,
        BLOCK_C=BLOCK_C,
        num_warps=ROW_NUM_WARPS,
    )

    reductions = torch.empty((3, hidden), device=arg0_1.device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(hidden, FINAL_BLOCK_C),)](
        partials[0],
        partials[1],
        partials[2],
        reductions[0],
        reductions[1],
        reductions[2],
        NUM_ROW_GROUPS=num_row_groups,
        C=hidden,
        BLOCK_GROUPS=FINAL_BLOCK_GROUPS,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=FINAL_NUM_WARPS,
    )

    return (
        grad,
        reductions[0],
        reductions[1],
        masked,
        masked.permute(1, 0),
        reductions[2],
    )
