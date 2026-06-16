"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GPT-J bf16 layer-norm-backward multi-output tail by converting the four bf16 producers to fp32, reconstructing the captured normalization producer `(bf16 + f32 - mean) * scale`, sharing each row's hidden-dimension reductions, storing the returned fp32 residual-add tensor plus the bf16-rounded view/transpose side outputs, and finalizing all three `[4096]` column reductions from row partials, whereas Inductor schedules the add chain, normalization producer, row-local reductions, residual epilogue, bf16 materialization, transpose view, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output reduction template that preserves explicit f32 and bf16 rounding boundaries while coordinating row reductions, side-output stores, and compatible column reductions; the fix is COOPERATIVE_SPLIT_K: add a row-tiled layer-norm-backward reduction template that keeps row scalars live, writes view-equivalent side outputs, and finalizes sibling column sums together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
CHANNELS = 4096
ROW_FACTOR = 4096.0
INV_ROW_FACTOR = 1.0 / 4096.0


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
def _store_row_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    weight_ptr,
    norm_bf16_ptr,
    norm_f32_ptr,
    mean_ptr,
    scale_ptr,
    residual_ptr,
    add_out_ptr,
    bf16_out_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    INV_ROW_FACTOR_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32)

    acc_x_norm = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_bf16_side = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS_
        offsets = rows[:, None] * CHANNELS_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = _add_rn(x, tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        x = _add_rn(x, tl.load(arg2_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
        x = _add_rn(x, tl.load(arg3_ptr + offsets, mask=mask, other=0.0).to(tl.float32))

        norm_src = tl.load(norm_bf16_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        norm_src = _add_rn(
            norm_src,
            tl.load(norm_f32_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        )
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        normed = _mul_rn(_sub_rn(norm_src, mean[:, None]), scale[:, None])

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = _mul_rn(x, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        weighted_norm = _mul_rn(weighted, normed)
        row_dot = tl.sum(tl.where(mask, weighted_norm, 0.0), axis=1)

        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(normed, row_dot[:, None]))
        scale_over_hidden = _mul_rn(scale, INV_ROW_FACTOR_)
        grad = _mul_rn(scale_over_hidden[:, None], centered)
        side = _add_rn(residual, grad)
        side_bf16 = side.to(tl.bfloat16)

        tl.store(add_out_ptr + offsets, side, mask=mask)
        tl.store(bf16_out_ptr + offsets, side_bf16, mask=mask)

        acc_x_norm += tl.sum(tl.where(mask, _mul_rn(x, normed), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_bf16_side += tl.sum(
            tl.where(mask, side_bf16.to(tl.float32), 0.0), axis=0
        )

    partial_base = group * 3 * CHANNELS_ + cols
    tl.store(partials_ptr + partial_base, acc_x_norm, mask=col_mask)
    tl.store(partials_ptr + partial_base + CHANNELS_, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * CHANNELS_, acc_bf16_side, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_bf16_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < CHANNELS_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * CHANNELS_ + cols[None, :]

    x_norm = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + CHANNELS_, mask=mask, other=0.0).to(
        tl.float32
    )
    bf16_side = tl.load(
        partials_ptr + offsets + 2 * CHANNELS_, mask=mask, other=0.0
    ).to(tl.float32)

    tl.store(out_x_norm_ptr + cols, tl.sum(x_norm, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    side_sum = tl.sum(bf16_side, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_bf16_sum_ptr + cols, side_sum, mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="85287631",
    ROWS_PER_GROUP=1,
    BLOCK_R=1,
    BLOCK_C=4096,
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
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        weight,
        norm_bf16,
        norm_f32,
        mean,
        scale,
        residual,
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    add_out = torch.empty_strided(
        (1, ROWS, CHANNELS),
        (ROWS * CHANNELS, CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        (ROWS, CHANNELS),
        (CHANNELS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty_strided(
        (num_groups, 3, CHANNELS),
        (3 * CHANNELS, CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    out_x_norm = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_bf16_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _store_row_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        weight,
        norm_bf16,
        norm_f32,
        mean,
        scale,
        residual,
        add_out,
        bf16_out,
        partials,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        ROW_FACTOR_=ROW_FACTOR,
        INV_ROW_FACTOR_=INV_ROW_FACTOR,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partials,
        out_x_norm,
        out_x,
        out_bf16_sum,
        NUM_GROUPS=num_groups,
        CHANNELS_=CHANNELS,
        GROUP_BLOCK=num_groups,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )
    return out_x_norm, out_x, add_out, bf16_out, bf16_out.permute(1, 0), out_bf16_sum
