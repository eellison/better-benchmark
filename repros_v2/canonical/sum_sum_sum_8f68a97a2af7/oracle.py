"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 SigLIP layer-norm-backward residual-add tail by row-tiling the fixed hidden size 768 source, sharing each row's hidden-dimension reductions across the bf16 residual-add output, returned transpose alias, and three compatible feature reductions, whereas Inductor schedules the row reductions, dependent bf16 add materialization, view/permute fanout, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output reduction template that preserves row-local scalars, explicit bf16 cast/add boundaries, required aliasing side outputs, and feature partials in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a row-tiled layer-norm-backward producer/finalizer schedule that writes the observable bf16 output and finalizes all sibling feature reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _produce_partials_kernel(
    x_ptr,
    weight_ptr,
    rhs_bf16_ptr,
    mean_ptr,
    inv_ptr,
    residual_bf16_ptr,
    out_bf16_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    ROW_GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_out = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_row in tl.range(0, ROW_GROUP):
        row = group * ROW_GROUP + local_row
        row_mask = row < ROWS
        mask = row_mask & col_mask
        offsets = row * CHANNELS + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs_src = tl.load(rhs_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        inv = tl.load(inv_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_bf16_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        weighted = _f32_mul(x, weight)
        rhs = _f32_mul(_f32_sub(rhs_src, mean), inv)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, _f32_mul(weighted, rhs), 0.0), axis=0)

        sub1 = _f32_sub(_f32_mul(weighted, 768.0), row_sum)
        sub2 = _f32_sub(sub1, _f32_mul(rhs, row_dot))
        grad = _f32_mul(_f32_div(inv, 768.0), sub2)
        grad_bf16 = grad.to(tl.bfloat16)
        out_bf16 = _f32_add(residual, grad_bf16.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_bf16_ptr + offsets, out_bf16, mask=mask)

        acc_x_rhs += tl.where(mask, _f32_mul(x, rhs), 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_out += tl.where(mask, out_bf16.to(tl.float32), 0.0)

    partial_offsets = group * CHANNELS + cols
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)
    tl.store(partial_out_ptr + partial_offsets, acc_out, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, BLOCK_G)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < CHANNELS)
    offsets = groups[:, None] * CHANNELS + cols[None, :]
    col_mask = cols < CHANNELS

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out_sum = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(
        out_sum_ptr + cols,
        tl.sum(out_sum, axis=0).to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )


def _next_power_of_2(x):
    return 1 << (x - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="b0021f14",
    ROW_GROUP=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
    num_warps_produce=8,
    num_warps_final=4,
)
def oracle_forward(
    inputs,
    *,
    ROW_GROUP: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps_produce: int,
    num_warps_final: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, *_ = inputs
    rows = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    num_groups = triton.cdiv(rows, ROW_GROUP)

    out_bf16 = torch.empty_strided(
        (rows, channels),
        (channels, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial_x_rhs = torch.empty(
        (num_groups, channels), device=arg0_1.device, dtype=torch.float32
    )
    partial_x = torch.empty(
        (num_groups, channels), device=arg0_1.device, dtype=torch.float32
    )
    partial_out = torch.empty(
        (num_groups, channels), device=arg0_1.device, dtype=torch.float32
    )
    sum_x_rhs = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    sum_x = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    sum_out = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)

    _produce_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out_bf16,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS=rows,
        CHANNELS=channels,
        ROW_GROUP=ROW_GROUP,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_produce,
        num_stages=3,
    )
    _finalize_partials_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        sum_x_rhs,
        sum_x,
        sum_out,
        NUM_GROUPS=num_groups,
        CHANNELS=channels,
        BLOCK_G=_next_power_of_2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )
    return sum_x_rhs, sum_x, out_bf16, out_bf16.permute(1, 0), sum_out
