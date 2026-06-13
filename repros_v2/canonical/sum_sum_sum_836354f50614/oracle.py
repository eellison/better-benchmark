"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full bf16-input transformer layer-norm-backward tail by row-tiling the shared `(arg0.to(f32) + arg1.to(f32)) + arg2.to(f32)` producer, keeping each row's hidden-dimension reductions local, preserving the captured literal 2560 factor/divisor in the epilogue, storing the returned fp32 gradient tensor, and cooperatively finalizing both returned hidden-column reductions from the same row-tile partials, whereas Inductor schedules the bf16 cast/add producer, row reductions, dense epilogue, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that preserves explicit bf16-to-fp32 rounding boundaries and captured scalar constants while coordinating row-local scalars, dependent epilogue stores, and compatible column partials; the fix is COOPERATIVE_SPLIT_K: add a row-tiled layer-norm-backward reduction plan that shares the cast/add producer and row summaries, emits the full returned tensor, and finalizes sibling column reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


NORM_FACTOR = 2560.0


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
def _div_rn(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_partials_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    weight_ptr,
    source_ptr,
    shift_ptr,
    scale_ptr,
    residual_ptr,
    out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NORM_FACTOR_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN
    norm_factor = tl.full((BLOCK_C,), NORM_FACTOR_, tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_m2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_row in tl.range(0, ROWS_PER_GROUP):
        row = group * ROWS_PER_GROUP + local_row
        row_mask = row < ROWS
        mask = row_mask & col_mask
        offsets = row * HIDDEN + cols

        x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = _add_rn(_add_rn(x0, x1), x2)

        source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        centered_source = _sub_rn(source, shift)
        m2 = _mul_rn(centered_source, scale)
        weighted = _mul_rn(x, weight)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, m2), 0.0), axis=0)

        centered_grad = _sub_rn(_mul_rn(weighted, norm_factor), row_sum)
        centered_grad = _sub_rn(centered_grad, _mul_rn(m2, row_dot))
        grad = _mul_rn(_div_rn(scale, norm_factor), centered_grad)
        tl.store(out_ptr + offsets, _add_rn(residual, grad), mask=mask)

        acc_x_m2 += tl.where(mask, _mul_rn(x, m2), 0.0)
        acc_x += tl.where(mask, x, 0.0)

    partial_base = group * 2 * HIDDEN + cols
    tl.store(partials_ptr + partial_base, acc_x_m2, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN, acc_x, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_m2_ptr,
    out_x_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, BLOCK_GROUPS)
    col_mask = cols < HIDDEN
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 2 * HIDDEN + cols[None, :]

    x_m2 = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + HIDDEN, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_m2_ptr + cols, tl.sum(x_m2, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 13570cda: (T([4096,2560], bf16), T([4096,2560], bf16), T([4096,2560], bf16), ...)
@oracle_impl(hardware="B200", point="13570cda", ROWS_PER_GROUP=8, BLOCK_C=4096, FINAL_BLOCK_C=8, num_warps=8)
# 637b22e7: (T([2048,2560], bf16), T([2048,2560], bf16), T([2048,2560], bf16), ...)
@oracle_impl(hardware="B200", point="637b22e7", ROWS_PER_GROUP=8, BLOCK_C=4096, FINAL_BLOCK_C=8, num_warps=8)
# c0772b17: (T([8192,1024], bf16), T([8192,1024], bf16), T([8192,1024], bf16), ...)
@oracle_impl(hardware="B200", point="c0772b17", ROWS_PER_GROUP=16, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
# 1cfd796c: (T([8192,1024], bf16), T([8192,1024], bf16), T([8192,1024], bf16), ...)
@oracle_impl(hardware="B200", point="1cfd796c", ROWS_PER_GROUP=16, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
# 2805f48b: (T([8192,768], bf16), T([8192,768], bf16), T([8192,768], bf16), ...)
@oracle_impl(hardware="B200", point="2805f48b", ROWS_PER_GROUP=16, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
# 83be941c: (T([16384,1024], bf16), T([16384,1024], bf16), T([16384,1024], bf16), ...)
@oracle_impl(hardware="B200", point="83be941c", ROWS_PER_GROUP=16, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
# 1247fcfd: (T([4096,1024], bf16), T([4096,1024], bf16), T([4096,1024], bf16), ...)
@oracle_impl(hardware="B200", point="1247fcfd", ROWS_PER_GROUP=16, BLOCK_C=1024, FINAL_BLOCK_C=8, num_warps=8)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    full_shape = _shape_tuple(shape_param_0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)

    out_x_m2 = torch.empty_strided((hidden,), (1,), device=arg0_1.device, dtype=torch.float32)
    out_x = torch.empty_strided((hidden,), (1,), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (num_groups, 2, hidden),
        (2 * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        out,
        partials,
        ROWS=rows,
        HIDDEN=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_C=BLOCK_C,
        NORM_FACTOR_=NORM_FACTOR,
        num_warps=num_warps,
    )

    block_groups = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_C),)](
        partials,
        out_x_m2,
        out_x,
        NUM_GROUPS=num_groups,
        HIDDEN=hidden,
        BLOCK_GROUPS=block_groups,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )

    return out_x_m2, out_x, out
