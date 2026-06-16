"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the
complete MobileBERT bf16/fp32 multi-output reduction scope with one token-tiled
producer plus a batch-partial finalizer, writing both returned bf16 side tensors
and their transpose aliases while accumulating the four direct fp32 column
reductions and the two reductions over bf16-rounded side tensors; Inductor
currently schedules the shared add/mul producers, explicit bf16 conversions,
view/transpose side outputs, duplicate scaled reductions, and sibling column
sums as separate generic regions over materialized intermediates; the fix is
COOPERATIVE_SPLIT_K: teach the scheduler to coordinate this row-token producer,
preserve dtype rounding boundaries, and finalize all compatible column partials
from one shared plan.
"""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
HIDDEN = 128


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
def _token_partials_and_sides_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    side0_ptr,
    side1_ptr,
    partial0_ptr,
    partial1_ptr,
    partial4_ptr,
    partial5_ptr,
    partial6_ptr,
    partial9_ptr,
    XNUMEL: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    HIDDEN_: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    token_offsets = tl.arange(0, RBLOCK)[None, :]
    xmask = xindex < XNUMEL

    hidden_index = xindex % HIDDEN_
    batch_index = xindex // HIDDEN_
    flat_row = batch_index * RBLOCK + token_offsets
    offsets = flat_row * HIDDEN_ + hidden_index

    arg0 = tl.load(arg0_ptr + offsets, mask=xmask, other=0.0).to(tl.float32)
    arg1 = tl.load(arg1_ptr + offsets, mask=xmask, other=0.0).to(tl.float32)
    arg2 = tl.load(arg2_ptr + offsets, mask=xmask, other=0.0).to(tl.float32)
    arg5 = tl.load(arg5_ptr + offsets, mask=xmask, other=0.0).to(tl.float32)
    scale3 = tl.load(arg3_ptr + hidden_index, mask=xmask, other=0.0).to(tl.float32)
    bias4 = tl.load(arg4_ptr + hidden_index, mask=xmask, other=0.0).to(tl.float32)
    scale6 = tl.load(arg6_ptr + hidden_index, mask=xmask, other=0.0).to(tl.float32)

    add = _add_rn(arg1, arg0)
    add_1 = _add_rn(_mul_rn(arg2, scale3), bias4)
    add_2 = _add_rn(arg5, add_1)
    mul_1 = _mul_rn(add, add_2)
    mul_2 = _mul_rn(add, scale6)
    side0 = mul_2.to(tl.bfloat16, fp_downcast_rounding="rtne")
    mul_3 = _mul_rn(mul_2, arg2)
    mul_4 = _mul_rn(mul_2, scale3)
    side1 = mul_4.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(side0_ptr + offsets, side0, mask=xmask)
    tl.store(side1_ptr + offsets, side1, mask=xmask)

    partial_offsets = batch_index * HIDDEN_ + hidden_index
    tl.store(partial0_ptr + partial_offsets, tl.sum(add, axis=1)[:, None], mask=xmask)
    tl.store(partial1_ptr + partial_offsets, tl.sum(mul_1, axis=1)[:, None], mask=xmask)
    tl.store(
        partial4_ptr + partial_offsets,
        tl.sum(side0.to(tl.float32), axis=1)[:, None],
        mask=xmask,
    )
    tl.store(partial5_ptr + partial_offsets, tl.sum(mul_2, axis=1)[:, None], mask=xmask)
    tl.store(partial6_ptr + partial_offsets, tl.sum(mul_3, axis=1)[:, None], mask=xmask)
    tl.store(
        partial9_ptr + partial_offsets,
        tl.sum(side1.to(tl.float32), axis=1)[:, None],
        mask=xmask,
    )


@triton.jit
def _finalize_batch_partials_kernel(
    partial0_ptr,
    partial1_ptr,
    partial4_ptr,
    partial5_ptr,
    partial6_ptr,
    partial9_ptr,
    out0_ptr,
    out1_ptr,
    out4_ptr,
    out5_ptr,
    out6_ptr,
    out9_ptr,
    HIDDEN_: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    col = tl.program_id(0)
    batch_offsets = tl.arange(0, BLOCK_BATCH)
    offsets = batch_offsets * HIDDEN_ + col

    acc0 = tl.load(partial0_ptr + offsets).to(tl.float32)
    acc1 = tl.load(partial1_ptr + offsets).to(tl.float32)
    acc4 = tl.load(partial4_ptr + offsets).to(tl.float32)
    acc5 = tl.load(partial5_ptr + offsets).to(tl.float32)
    acc6 = tl.load(partial6_ptr + offsets).to(tl.float32)
    acc9 = tl.load(partial9_ptr + offsets).to(tl.float32)

    tl.store(out0_ptr + col, tl.sum(acc0, axis=0))
    tl.store(out1_ptr + col, tl.sum(acc1, axis=0))
    tl.store(
        out4_ptr + col,
        tl.sum(acc4, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
    )
    tl.store(out5_ptr + col, tl.sum(acc5, axis=0))
    tl.store(out6_ptr + col, tl.sum(acc6, axis=0))
    tl.store(
        out9_ptr + col,
        tl.sum(acc9, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# (T([32768,128], bf16), T([256,128,128], f32), T([32768,128], bf16), ...)
@oracle_impl(hardware="B200", point="fd972c32", XBLOCK=32, num_warps=8, final_warps=8)
def oracle_forward(inputs, *, XBLOCK: int, num_warps: int, final_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        _shape_param_0,
        shape_param_1,
        _shape_param_2,
        _shape_param_3,
        shape_param_4,
        shape_param_5,
        shape_param_6,
        shape_param_7,
        shape_param_8,
        shape_param_9,
        shape_param_10,
    ) = inputs

    sum_shape = _shape_tuple(shape_param_1)
    side0_shape = _shape_tuple(shape_param_5)
    side1_shape = _shape_tuple(shape_param_9)
    hidden = int(arg0_1.shape[1])

    side0 = torch.empty_strided(
        side0_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    side1 = torch.empty_strided(
        side1_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    partial0 = torch.empty_strided((BATCH, hidden), (hidden, 1), device=arg0_1.device, dtype=torch.float32)
    partial1 = torch.empty_strided((BATCH, hidden), (hidden, 1), device=arg0_1.device, dtype=torch.float32)
    partial4 = torch.empty_strided((BATCH, hidden), (hidden, 1), device=arg0_1.device, dtype=torch.float32)
    partial5 = torch.empty_strided((BATCH, hidden), (hidden, 1), device=arg0_1.device, dtype=torch.float32)
    partial6 = torch.empty_strided((BATCH, hidden), (hidden, 1), device=arg0_1.device, dtype=torch.float32)
    partial9 = torch.empty_strided((BATCH, hidden), (hidden, 1), device=arg0_1.device, dtype=torch.float32)

    out0 = torch.empty_strided(sum_shape, (1,), device=arg0_1.device, dtype=torch.float32)
    out1 = torch.empty_strided(_shape_tuple(shape_param_4), (1,), device=arg0_1.device, dtype=torch.float32)
    out4 = torch.empty_strided(_shape_tuple(shape_param_6), (1,), device=arg0_1.device, dtype=torch.float32)
    out5 = torch.empty_strided(_shape_tuple(shape_param_7), (1,), device=arg0_1.device, dtype=torch.float32)
    out6 = torch.empty_strided(_shape_tuple(shape_param_8), (1,), device=arg0_1.device, dtype=torch.float32)
    out9 = torch.empty_strided(_shape_tuple(shape_param_10), (1,), device=arg0_1.device, dtype=torch.float32)

    xnumel = BATCH * hidden
    _token_partials_and_sides_kernel[(triton.cdiv(xnumel, XBLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        side0,
        side1,
        partial0,
        partial1,
        partial4,
        partial5,
        partial6,
        partial9,
        XNUMEL=xnumel,
        XBLOCK=XBLOCK,
        RBLOCK=TOKENS,
        HIDDEN_=hidden,
        num_warps=num_warps,
    )
    _finalize_batch_partials_kernel[(hidden,)](
        partial0,
        partial1,
        partial4,
        partial5,
        partial6,
        partial9,
        out0,
        out1,
        out4,
        out5,
        out6,
        out9,
        HIDDEN_=hidden,
        BLOCK_BATCH=BATCH,
        num_warps=final_warps,
    )

    return (
        out0,
        out1,
        side0,
        side0.permute(1, 0),
        out4,
        out5,
        out6,
        side1,
        side1.permute(1, 0),
        out9,
    )
