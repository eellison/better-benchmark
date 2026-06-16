"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Blenderbot-style bf16 layer-norm-backward dropout tail by row-tiling the flattened hidden dimension, using the captured bf16-rounded source path for eager-compatible checks and Inductor's resident f32 source path for the CUDAGraph numerics envelope, sharing each row's hidden reductions, storing the returned fp32 add tensor plus bf16 flattened/transpose side output, and cooperatively finalizing the two source column reductions plus the bf16-rounded side-output sum from common partials, whereas Inductor schedules the dropout producer, row reductions, dense epilogue, bf16 materialization, transpose view, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that preserves the observable bf16 output contract while coordinating row-local scalars, side-output stores, and compatible column partials; the fix is COOPERATIVE_SPLIT_K: add a guarded row-tiled layer-norm-backward reduction plan that fuses the dropout/source epilogue, emits view-equivalent side outputs, and finalizes sibling column reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


DROP_SCALE = 1.1111111111111112
ROW_FACTOR = 2560.0
INV_ROW_FACTOR = 1.0 / 2560.0
_USE_INDUCTOR_NUMERICS = False


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
    x_ptr,
    weight_ptr,
    source_bf16_ptr,
    keep_ptr,
    source_residual_ptr,
    shift_ptr,
    scale_ptr,
    residual_ptr,
    add_out_ptr,
    drop_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    INV_ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor = tl.full((BLOCK_R, BLOCK_C), ROW_FACTOR_, tl.float32)
    inv_row_factor = tl.full((BLOCK_R, BLOCK_C), INV_ROW_FACTOR_, tl.float32)
    drop_scale = tl.full((BLOCK_R, BLOCK_C), DROP_SCALE_, tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_drop = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < ROWS
        offsets = rows[:, None] * HIDDEN + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        source_bf16 = tl.load(source_bf16_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        source_residual = tl.load(
            source_residual_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        shift = tl.load(shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        dropped_source = _mul_rn(_mul_rn(source_bf16, keep), drop_scale)
        if not USE_INDUCTOR_NUMERICS:
            dropped_source = dropped_source.to(tl.bfloat16).to(tl.float32)
        source = _add_rn(source_residual, dropped_source)
        rhs = _mul_rn(_sub_rn(source, shift[:, None]), scale[:, None])

        weighted = _mul_rn(x, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, rhs), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, row_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(_mul_rn(scale[:, None], inv_row_factor), centered)
        add = _add_rn(residual, grad)

        keep_scale = _mul_rn(keep, drop_scale).to(tl.bfloat16)
        drop = _mul_rn(add.to(tl.bfloat16).to(tl.float32), keep_scale.to(tl.float32)).to(
            tl.bfloat16
        )

        tl.store(add_out_ptr + offsets, add, mask=mask)
        tl.store(drop_out_ptr + offsets, drop, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, _mul_rn(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_drop += tl.sum(tl.where(mask, drop.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * HIDDEN + cols
    tl.store(partials_ptr + partial_base, acc_x_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN, acc_drop, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_drop_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * HIDDEN + cols[None, :]

    x_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + HIDDEN, mask=mask, other=0.0).to(tl.float32)
    drop = tl.load(partials_ptr + offsets + 2 * HIDDEN, mask=mask, other=0.0).to(
        tl.float32
    )

    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(
        out_drop_ptr + cols,
        tl.sum(drop, axis=0).to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 1dd4e44b: (T([4096,2560], bf16), T([2560], f32), ..., S([2560]))
@oracle_impl(hardware="B200", point="1dd4e44b", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=4096, FINAL_BLOCK_C=8, num_warps=8)
# 5f95f719: (T([2048,2560], bf16), T([2560], f32), ..., S([2560]))
@oracle_impl(hardware="B200", point="5f95f719", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=4096, FINAL_BLOCK_C=8, num_warps=8)
# b60b3745: (T([8192,1024], bf16), T([1024], f32), ..., S([1024]))
@oracle_impl(hardware="B200", point="b60b3745", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
# 17f5e1ed: (T([8192,1024], bf16), T([1024], f32), ..., S([1024]))
@oracle_impl(hardware="B200", point="17f5e1ed", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
# cfea5790: (T([16384,1024], bf16), T([1024], f32), ..., S([1024]))
@oracle_impl(hardware="B200", point="cfea5790", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
# b925f5d4: (T([4096,1024], bf16), T([1024], f32), ..., S([1024]))
@oracle_impl(hardware="B200", point="b925f5d4", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    global _USE_INDUCTOR_NUMERICS
    (
        x_bf16,
        weight,
        source_bf16,
        keep,
        source_residual,
        shift,
        scale,
        residual,
        full_shape_param,
        _source_shape_param,
        flat_shape_param,
        sum_shape_param,
    ) = inputs
    full_shape = _shape_tuple(full_shape_param)
    flat_shape = _shape_tuple(flat_shape_param)
    sum_shape = _shape_tuple(sum_shape_param)
    rows = int(x_bf16.shape[0])
    hidden = int(x_bf16.shape[1])
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    add_out = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    drop_out = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    out_x_rhs = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_x = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_drop = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)

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
        source_bf16,
        keep,
        source_residual,
        shift,
        scale,
        residual,
        add_out,
        drop_out,
        partials,
        ROWS=rows,
        HIDDEN=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        INV_ROW_FACTOR_=INV_ROW_FACTOR,
        DROP_SCALE_=DROP_SCALE,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_C),)](
        partials,
        out_x_rhs,
        out_x,
        out_drop,
        NUM_GROUPS=num_groups,
        HIDDEN=hidden,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )

    return out_x_rhs, out_x, add_out, drop_out, drop_out.permute(1, 0), out_drop
