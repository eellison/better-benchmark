"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full MobileBERT bf16 layer-norm-backward return tuple with one row-tiled producer that shares the per-row hidden reductions, writes the returned bf16 masked-gradient tensor and its transpose alias directly, and cooperatively finalizes the three returned `[512]` column reductions from common partials, whereas Inductor schedules the row-local reductions, bf16 mask/materialization, transpose view producer, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that preserves explicit bf16/fp32 conversion boundaries while coordinating row reductions, side-output stores, and compatible column partials; the fix is COOPERATIVE_SPLIT_K: add a row-tiled layer-norm-backward reduction plan that keeps row summaries live, sinks view-only layout algebra into output planning, and finalizes all sibling column reductions together."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 32768
HIDDEN = 512
ROW_FACTOR = 512.0
INV_ROW_FACTOR = 1.0 / 512.0


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
    relu_input_ptr,
    center_ptr,
    scale_ptr,
    full_out_ptr,
    masked_out_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    INV_ROW_FACTOR_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32)
    inv_row_factor = tl.full((BLOCK_R, 1), INV_ROW_FACTOR_, tl.float32)

    acc_x_norm = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_C,), dtype=tl.float32)

    if group == 0:
        tl.store(full_out_ptr, tl.full((), 0.0, tl.float32))

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        relu_input = tl.load(relu_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        center = tl.load(center_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        relu = tl.maximum(relu_input, 0.0)
        norm = _mul_rn(_sub_rn(relu, center[:, None]), scale[:, None])
        weighted = _mul_rn(x, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, norm), 0.0), axis=1)

        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(norm, row_dot[:, None]))
        grad = _mul_rn(_mul_rn(scale[:, None], inv_row_factor), centered)
        grad_bf16_f32 = grad.to(tl.bfloat16).to(tl.float32)
        masked = tl.where(relu <= 0.0, 0.0, grad_bf16_f32)

        tl.store(masked_out_ptr + offsets, masked, mask=mask)

        acc_x_norm += tl.sum(tl.where(mask, _mul_rn(x, norm), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_masked += tl.sum(tl.where(mask, masked, 0.0), axis=0)

    partial_base = group * 3 * HIDDEN_ + cols
    tl.store(partials_ptr + partial_base, acc_x_norm, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN_, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN_, acc_masked, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_masked_sum_ptr,
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

    x_norm = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(tl.float32)
    masked = tl.load(partials_ptr + offsets + 2 * HIDDEN_, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_norm_ptr + cols, tl.sum(x_norm, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(
        out_masked_sum_ptr + cols,
        tl.sum(masked, axis=0).to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="81eeb2b9",
    ROWS_PER_GROUP=16,
    BLOCK_R=1,
    BLOCK_C=512,
    FINAL_BLOCK_C=8,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    (
        x_bf16,
        weight,
        relu_input_bf16,
        center,
        scale,
        _shape0,
        _shape1,
        masked_shape_param,
        sum_shape_param,
    ) = inputs
    masked_shape = _shape_tuple(masked_shape_param)
    sum_shape = _shape_tuple(sum_shape_param)
    rows = int(x_bf16.shape[0])
    hidden = int(x_bf16.shape[1])

    full = torch.empty_strided((), (), device=x_bf16.device, dtype=torch.bfloat16)
    masked = torch.empty_strided(
        masked_shape,
        (hidden, 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    out_x_norm = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_x = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_masked_sum = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)

    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, hidden),
        (3 * hidden, hidden, 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        x_bf16,
        weight,
        relu_input_bf16,
        center,
        scale,
        full,
        masked,
        partials,
        ROWS_=rows,
        HIDDEN_=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        INV_ROW_FACTOR_=INV_ROW_FACTOR,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_C),)](
        partials,
        out_x_norm,
        out_x,
        out_masked_sum,
        NUM_GROUPS=num_groups,
        HIDDEN_=hidden,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )

    return out_x_norm, out_x, full, masked, masked.permute(1, 0), out_masked_sum
