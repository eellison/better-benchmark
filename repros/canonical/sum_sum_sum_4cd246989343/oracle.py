"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Google FNet layer-norm/dropout-backward return tuple by row-tiling the selected-real `[32,512,768]` producer, storing both the unmasked gradient and the masked flat/transpose side outputs, and cooperatively accumulating the two pre-mask `[768]` column sums plus the masked-gradient `[768]` column sum from shared row partials, whereas Inductor schedules the selected-view add, row-local reductions, dropout-mask multiply, view/permute side output, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that combines row-local layernorm-backward scalars, required layout side outputs, and multiple compatible column accumulators in one coordinated producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer/finalizer schedule that reads selected-view inputs directly, sinks the required side-output stores, and finalizes all compatible column reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
DROP_SCALE = 1.1111111111111112
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
    real_pair_ptr,
    residual_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    keep_ptr,
    grad_out_ptr,
    masked_out_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32)
    drop_scale = tl.full((BLOCK_R, BLOCK_C), DROP_SCALE_, tl.float32)

    acc_add_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_add = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        selected = tl.load(real_pair_ptr + offsets * 2, mask=mask, other=0.0).to(
            tl.float32
        )
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)

        add = _add_rn(residual, selected)
        weighted = _mul_rn(add, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, rhs), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(row_scale[:, None], centered)
        keep_scale = _mul_rn(keep, drop_scale)
        masked_grad = _mul_rn(grad, keep_scale)

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(masked_out_ptr + offsets, masked_grad, mask=mask)

        acc_add_rhs += tl.sum(tl.where(mask, _mul_rn(add, rhs), 0.0), axis=0)
        acc_add += tl.sum(tl.where(mask, add, 0.0), axis=0)
        acc_masked += tl.sum(tl.where(mask, masked_grad, 0.0), axis=0)

    partial_base = group * 3 * HIDDEN_ + cols
    tl.store(partials_ptr + partial_base, acc_add_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN_, acc_add, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN_, acc_masked, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_add_rhs_ptr,
    out_add_ptr,
    out_masked_ptr,
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

    add_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(
        tl.float32
    )
    masked = tl.load(partials_ptr + offsets + 2 * HIDDEN_, mask=mask, other=0.0).to(
        tl.float32
    )

    tl.store(out_add_rhs_ptr + cols, tl.sum(add_rhs, axis=0), mask=col_mask)
    tl.store(out_add_ptr + cols, tl.sum(add, axis=0), mask=col_mask)
    tl.store(out_masked_ptr + cols, tl.sum(masked, axis=0), mask=col_mask)


@oracle_impl(
    hardware="B200",
    point="3847e61c",
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
    (
        real_pair,
        residual,
        weight,
        rhs,
        row_scale,
        keep,
        _flat_shape,
        _sum_shape,
    ) = inputs

    grad_out = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=residual.device,
        dtype=torch.float32,
    )
    masked_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=residual.device,
        dtype=torch.float32,
    )
    out_add_rhs = torch.empty_strided(
        (HIDDEN,), (1,), device=residual.device, dtype=torch.float32
    )
    out_add = torch.empty_strided(
        (HIDDEN,), (1,), device=residual.device, dtype=torch.float32
    )
    out_masked = torch.empty_strided(
        (HIDDEN,), (1,), device=residual.device, dtype=torch.float32
    )

    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    partials = torch.empty_strided(
        (num_groups, 3, HIDDEN),
        (3 * HIDDEN, HIDDEN, 1),
        device=residual.device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        real_pair,
        residual,
        weight,
        rhs,
        row_scale,
        keep,
        grad_out,
        masked_out,
        partials,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        DROP_SCALE_=DROP_SCALE,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_C),)](
        partials,
        out_add_rhs,
        out_add,
        out_masked,
        NUM_GROUPS=num_groups,
        HIDDEN_=HIDDEN,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    return grad_out, out_add_rhs, out_add, masked_out, masked_out.permute(1, 0), out_masked
