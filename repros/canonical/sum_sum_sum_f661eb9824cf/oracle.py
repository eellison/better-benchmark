"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GPT-2 sequence-classification bf16 layer-norm-backward/dropout scope by row-tiling the fixed `[8192, 768]` producer, keeping each row's hidden-dimension reductions live through the fp32 input-gradient epilogue, materializing the returned bf16 dropout buffer with the explicit bf16 cast/product boundaries, and cooperatively finalizing the two fp32 column sums plus the bf16-rounded dropout column sum, whereas Inductor lowers the row reductions, dense fp32 gradient store, bf16 dropout materialization, aliasing view, and sibling column reductions as separate generic pointwise and reduction regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that coordinates row-local layer-norm-backward scalars with a live bf16 dropout side output and multiple compatible column reductions while preserving dtype boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded row-tiled layer-norm-backward/dropout schedule that writes the observable dense outputs once and finalizes sibling column partials directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
ROW_FACTOR = 768.0
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
def _row_partials_kernel(
    x_ptr,
    weight_ptr,
    normed_ptr,
    row_scale_ptr,
    keep_ptr,
    grad_out_ptr,
    drop_out_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_x_normed = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_drop = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for row_offset in tl.static_range(0, ROWS_PER_GROUP):
        row = group * ROWS_PER_GROUP + row_offset
        row_active = row < ROWS_
        offsets = row * HIDDEN_ + cols
        mask = row_active & col_mask

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row_active, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)

        weighted = x * weight
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        weighted_normed = weighted * normed
        row_dot = tl.sum(tl.where(mask, weighted_normed, 0.0), axis=0)
        centered = (weighted * ROW_FACTOR_ - row_sum) - normed * row_dot
        grad = row_scale * centered

        keep_scale_bf16 = _mul_rn(keep, DROP_SCALE_).to(tl.bfloat16)
        grad_bf16 = grad.to(tl.bfloat16)
        drop_bf16 = _mul_rn(
            grad_bf16.to(tl.float32),
            keep_scale_bf16.to(tl.float32),
        ).to(tl.bfloat16)
        store_drop_bf16 = (grad * keep_scale_bf16.to(tl.float32)).to(tl.bfloat16)

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(drop_out_ptr + offsets, store_drop_bf16, mask=mask)

        acc_x_normed += tl.where(mask, x * normed, 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_drop += tl.where(mask, drop_bf16.to(tl.float32), 0.0)

    partial_base = group * 3 * HIDDEN_ + cols
    tl.store(partials_ptr + partial_base, acc_x_normed, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN_, acc_x, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN_, acc_drop, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_normed_ptr,
    out_x_ptr,
    out_drop_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * HIDDEN_ + cols[None, :]

    x_normed = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(tl.float32)
    drop = tl.load(partials_ptr + offsets + 2 * HIDDEN_, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_x_normed_ptr + cols, tl.sum(x_normed, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_drop_ptr + cols, tl.sum(drop, axis=0).to(tl.bfloat16).to(tl.float32), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="d5cd1dee",
    ROWS_PER_GROUP=4,
    BLOCK_H=1024,
    FINAL_BLOCK_H=8,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_H: int,
    FINAL_BLOCK_H: int,
    row_warps: int,
    final_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    del _shape_param_0

    device = arg0.device
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])
    num_groups = triton.cdiv(rows, ROWS_PER_GROUP)

    grad_out = torch.empty_strided(
        (8, 1024, hidden),
        (1024 * hidden, hidden, 1),
        device=device,
        dtype=torch.float32,
    )
    drop_out = torch.empty_strided(
        _shape_tuple(_shape_param_1),
        (hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_x_normed = torch.empty_strided(
        _shape_tuple(_shape_param_2),
        (1,),
        device=device,
        dtype=torch.float32,
    )
    out_x = torch.empty_strided(
        _shape_tuple(_shape_param_2),
        (1,),
        device=device,
        dtype=torch.float32,
    )
    out_drop = torch.empty_strided(
        _shape_tuple(_shape_param_2),
        (1,),
        device=device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (num_groups, 3, hidden),
        (3 * hidden, hidden, 1),
        device=device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        grad_out,
        drop_out,
        partials,
        ROWS_=rows,
        HIDDEN_=hidden,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_H=BLOCK_H,
        ROW_FACTOR_=ROW_FACTOR,
        DROP_SCALE_=DROP_SCALE,
        num_warps=row_warps,
    )
    _finalize_partials_kernel[(triton.cdiv(hidden, FINAL_BLOCK_H),)](
        partials,
        out_x_normed,
        out_x,
        out_drop,
        NUM_GROUPS=num_groups,
        HIDDEN_=hidden,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=final_warps,
    )

    return grad_out, out_x_normed, out_x, drop_out, out_drop
