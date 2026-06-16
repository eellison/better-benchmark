"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Electra bf16/fp32 layer-norm-backward tail by sharing each token row's hidden-dimension reductions, writing the returned bf16 gradient tensor plus its dropout-masked view/permute aliases, and cooperatively finalizing the two source column reductions plus the bf16-rounded side-output sum from common row-tile partials, whereas Inductor currently schedules the add producer, row-local reductions, bf16/dropout epilogue, layout aliases, and sibling column reductions as separate generic regions over materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, exact bf16 rounding boundaries, layout side outputs, and compatible column partials in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled layer-norm-backward reduction template that fuses shared producers, writes view-equivalent side outputs, and finalizes all sibling column reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROW_FACTOR = 256.0
INV_ROW_FACTOR = 0.00390625
DROP_SCALE = 1.1111111111111112


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
def _store_partials_kernel(
    bf16_residual_ptr,
    f32_residual_ptr,
    weight_ptr,
    norm_source_ptr,
    mean_ptr,
    scale_ptr,
    mask_ptr,
    grad_bf16_out_ptr,
    side_bf16_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    INV_ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_side = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * CHANNELS + cols[None, :]

        x = tl.load(f32_residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = _add_rn(
            x,
            tl.load(bf16_residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        )
        norm_source = tl.load(norm_source_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        keep = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.float32)

        rhs = _mul_rn(_sub_rn(norm_source, mean[:, None]), scale[:, None])
        weighted = _mul_rn(x, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, rhs), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, ROW_FACTOR_), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        row_scale = _mul_rn(scale, INV_ROW_FACTOR_)
        grad = _mul_rn(row_scale[:, None], centered)
        grad = tl.where(mask, grad, 0.0)

        grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne")
        keep_scale = _mul_rn(keep, DROP_SCALE_).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        )
        side_bf16 = _mul_rn(grad_bf16.to(tl.float32), keep_scale.to(tl.float32)).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        )

        tl.store(grad_bf16_out_ptr + offsets, grad_bf16, mask=mask)
        tl.store(side_bf16_out_ptr + offsets, side_bf16, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, _mul_rn(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_side += tl.sum(tl.where(mask, side_bf16.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * CHANNELS + cols
    tl.store(partials_ptr + partial_base, acc_x_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + CHANNELS, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * CHANNELS, acc_side, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_side_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < CHANNELS)
    offsets = groups[:, None] * 3 * CHANNELS + cols[None, :]
    col_mask = cols < CHANNELS

    x_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + CHANNELS, mask=mask, other=0.0).to(tl.float32)
    side = tl.load(partials_ptr + offsets + 2 * CHANNELS, mask=mask, other=0.0).to(
        tl.float32
    )

    side_sum = tl.sum(side, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(
        tl.float32
    )
    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_side_sum_ptr + cols, side_sum, mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="b0f8aad0",
    ROWS_PER_GROUP=128,
    BLOCK_R=32,
    BLOCK_C=256,
    FINAL_BLOCK_C=8,
    num_warps=4,
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
        *_shape_params,
    ) = inputs
    batch = int(arg1_1.shape[0])
    seq = int(arg1_1.shape[1])
    channels = int(arg1_1.shape[2])
    rows = batch * seq
    full_shape = (batch, seq, channels)
    flat_shape = (rows, channels)

    grad_bf16 = torch.empty_strided(
        full_shape,
        (seq * channels, channels, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    side_bf16 = torch.empty_strided(
        flat_shape,
        (channels, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out_x_rhs = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    out_x = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)
    out_side_sum = torch.empty((channels,), device=arg1_1.device, dtype=torch.float32)

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, channels),
        (3 * channels, channels, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _store_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        grad_bf16,
        side_bf16,
        partials,
        ROWS=rows,
        CHANNELS=channels,
        ROW_FACTOR_=ROW_FACTOR,
        INV_ROW_FACTOR_=INV_ROW_FACTOR,
        DROP_SCALE_=DROP_SCALE,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
        partials,
        out_x_rhs,
        out_x,
        out_side_sum,
        NUM_GROUPS=num_groups,
        CHANNELS=channels,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )
    return out_x_rhs, out_x, grad_bf16, side_bf16, side_bf16.permute(1, 0), out_side_sum
