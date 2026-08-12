"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ConvBERT bf16 reshape/permute tail by row-tiling the shared `[16384, 384]` producer, preserving the explicit bf16 round after the fp32 bias add, both bf16 multiply/add side outputs, the returned transpose views, and the two `[384]` column reductions from common row partials, whereas Inductor schedules the layout clone, sibling pointwise producers, returned view materializations, and reductions as separate generic regions over replayed or materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates layout-changing side outputs, compatible column reductions, and bf16 dtype boundaries from one producer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled multi-output reduction plan that writes required side outputs in final layout while co-finalizing sibling column sums."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 384
SEQ = 512
ARG0_BATCH_STRIDE = 196608
ARG0_SEQ_STRIDE = 384
ARG0_HEAD_STRIDE = 64
ARG2_BATCH_STRIDE = 196608
ARG2_HIDDEN_STRIDE = 512


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _convbert_row_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    out_add_ptr,
    out_mul_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    ARG0_BATCH_STRIDE_: tl.constexpr,
    ARG0_SEQ_STRIDE_: tl.constexpr,
    ARG0_HEAD_STRIDE_: tl.constexpr,
    ARG2_BATCH_STRIDE_: tl.constexpr,
    ARG2_HIDDEN_STRIDE_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_group = tl.program_id(1)
    row_offsets = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_group * BLOCK_H + tl.arange(0, BLOCK_H)
    rows = row_offsets[:, None]
    hidden = cols[None, :]
    mask = (rows < ROWS_) & (hidden < HIDDEN_)

    batch = rows // SEQ_
    seq = rows - batch * SEQ_
    head = hidden // 64
    dim = hidden - head * 64

    flat_offsets = rows * HIDDEN_ + hidden
    arg0_offsets = (
        batch * ARG0_BATCH_STRIDE_
        + seq * ARG0_SEQ_STRIDE_
        + head * ARG0_HEAD_STRIDE_
        + dim
    )
    arg2_offsets = batch * ARG2_BATCH_STRIDE_ + hidden * ARG2_HIDDEN_STRIDE_ + seq

    view = tl.load(arg1_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(arg3_ptr + hidden, mask=hidden < HIDDEN_, other=0.0).to(tl.float32)
    rhs = _add_rn_f32(
        tl.load(arg2_ptr + arg2_offsets, mask=mask, other=0.0).to(tl.float32),
        bias,
    ).to(tl.bfloat16)

    mul = _mul_rn_f32(view, rhs.to(tl.float32)).to(tl.bfloat16)
    clone_value = tl.load(arg0_ptr + arg0_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    add_out = _add_rn_f32(mul.to(tl.float32), clone_value).to(tl.bfloat16)

    arg4 = tl.load(arg4_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
    mul_out = _mul_rn_f32(view, arg4).to(tl.bfloat16)

    tl.store(out_add_ptr + flat_offsets, add_out, mask=mask)
    tl.store(out_mul_ptr + flat_offsets, mul_out, mask=mask)

    col_mask = cols < HIDDEN_
    partial_base = row_group * 2 * HIDDEN_ + cols
    tl.store(
        partials_ptr + partial_base,
        tl.sum(tl.where(mask, add_out.to(tl.float32), 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partials_ptr + partial_base + HIDDEN_,
        tl.sum(tl.where(mask, mul_out.to(tl.float32), 0.0), axis=0),
        mask=col_mask,
    )


@triton.jit
def _convbert_finalize_kernel(
    partials_ptr,
    sum_add_ptr,
    sum_mul_ptr,
    HIDDEN_: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    groups = tl.arange(0, GROUP_BLOCK)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < HIDDEN_)
    offsets = groups[:, None] * 2 * HIDDEN_ + cols[None, :]

    add_partials = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mul_partials = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(
        tl.float32
    )

    col_mask = cols < HIDDEN_
    tl.store(
        sum_add_ptr + cols,
        tl.sum(add_partials, axis=0).to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )
    tl.store(sum_mul_ptr + cols, tl.sum(mul_partials, axis=0), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 0c475f9a: (T([32,6,512,64], bf16, stride=(196608,64,384,1)), T([16384,384], bf16), T([32,384,512], bf16), T([384,1], f32), T([32,512,384], bf16), ...)
@oracle_impl(
    hardware="B200",
    point="0c475f9a",
    ROW_BLOCK=64,
    BLOCK_H=128,
    FINAL_BLOCK_H=16,
    num_warps=4,
)
def oracle_forward(inputs, *, ROW_BLOCK, BLOCK_H, FINAL_BLOCK_H, num_warps):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        _shape_param_0,
        _shape_param_1,
        shape_view2,
        shape_sum0,
        shape_sum1,
    ) = inputs

    out_add = torch.empty_strided(
        _shape_tuple(shape_view2),
        (HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out_mul = torch.empty_strided(
        _shape_tuple(_shape_param_0),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    sum_add = torch.empty_strided(
        _shape_tuple(shape_sum0),
        (1,),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    sum_mul = torch.empty_strided(
        _shape_tuple(shape_sum1),
        (1, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    num_groups = triton.cdiv(ROWS, ROW_BLOCK)
    num_col_groups = triton.cdiv(HIDDEN, BLOCK_H)
    partials = torch.empty_strided(
        (num_groups, 2, HIDDEN),
        (2 * HIDDEN, HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _convbert_row_partials_kernel[(num_groups, num_col_groups)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out_add,
        out_mul,
        partials,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        ARG0_BATCH_STRIDE_=ARG0_BATCH_STRIDE,
        ARG0_SEQ_STRIDE_=ARG0_SEQ_STRIDE,
        ARG0_HEAD_STRIDE_=ARG0_HEAD_STRIDE,
        ARG2_BATCH_STRIDE_=ARG2_BATCH_STRIDE,
        ARG2_HIDDEN_STRIDE_=ARG2_HIDDEN_STRIDE,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    _convbert_finalize_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_H),)](
        partials,
        sum_add,
        sum_mul,
        HIDDEN_=HIDDEN,
        NUM_GROUPS=num_groups,
        GROUP_BLOCK=triton.next_power_of_2(num_groups),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=8,
    )

    return out_add, out_add.permute(1, 0), sum_add, out_mul.permute(0, 2, 1), sum_mul
