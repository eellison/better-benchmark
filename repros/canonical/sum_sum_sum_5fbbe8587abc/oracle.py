"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileBERT bf16/fp32 multi-output reduction scope in one row-tiled Triton pass plus a partial-sum finalizer, including the bf16-to-fp32 add producer, the returned fp32 scaled side tensor, the bf16-rounded side tensor and transpose view, and all three `[128]` column reductions, whereas Inductor currently schedules the shared add, sibling reductions, fp32 side-output materialization, bf16 conversion, transpose view, and final bf16-side reduction as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps explicit dtype-conversion boundaries, layout-producing side outputs, and compatible column partials in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add multi-output reduction scheduling that shares the producer, writes required side-output layouts directly, and finalizes all compatible column sums from common row-tile partials."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
HIDDEN = 128
ROWS = BATCH * TOKENS


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
def _token_partials_and_side_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    side_f32_ptr,
    side_bf16_ptr,
    partial0_ptr,
    partial1_ptr,
    partial3_ptr,
    arg0_s0: tl.constexpr,
    arg0_s1: tl.constexpr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg2_s0: tl.constexpr,
    arg2_s1: tl.constexpr,
    arg2_s2: tl.constexpr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    xnumel: tl.constexpr,
    xblock: tl.constexpr,
    rblock: tl.constexpr,
):
    xindex = tl.program_id(0) * xblock + tl.arange(0, xblock)[:, None]
    token_offsets = tl.arange(0, rblock)[None, :]

    hidden_index = xindex % hidden
    batch_index = xindex // hidden
    flat_row = batch_index * tokens + token_offsets

    arg0_offsets = flat_row * arg0_s0 + hidden_index * arg0_s1
    arg1_offsets = (
        batch_index * arg1_s0
        + token_offsets * arg1_s1
        + hidden_index * arg1_s2
    )
    arg2_offsets = (
        batch_index * arg2_s0
        + token_offsets * arg2_s1
        + hidden_index * arg2_s2
    )

    add = _add_rn(
        tl.load(arg1_ptr + arg1_offsets).to(tl.float32),
        tl.load(arg0_ptr + arg0_offsets).to(tl.float32),
    )
    arg2 = tl.load(arg2_ptr + arg2_offsets).to(tl.float32)
    scale = tl.load(arg3_ptr + hidden_index).to(tl.float32)
    side = _mul_rn(add, scale)
    side_bf16 = side.to(tl.bfloat16)

    flat_offsets = flat_row * hidden + hidden_index
    tl.store(side_f32_ptr + flat_offsets, side)
    tl.store(side_bf16_ptr + flat_offsets, side_bf16)

    partial_offsets = batch_index * hidden + hidden_index
    tl.store(partial0_ptr + partial_offsets, tl.sum(add, axis=1)[:, None])
    tl.store(
        partial1_ptr + partial_offsets,
        tl.sum(_mul_rn(add, arg2), axis=1)[:, None],
    )
    tl.store(
        partial3_ptr + partial_offsets,
        tl.sum(side_bf16.to(tl.float32), axis=1)[:, None],
    )


@triton.jit
def _finalize_batch_partials_kernel(
    partial0_ptr,
    partial1_ptr,
    partial3_ptr,
    out0_ptr,
    out1_ptr,
    out3_ptr,
    hidden: tl.constexpr,
    block_batch: tl.constexpr,
):
    col = tl.program_id(0)
    batch_offsets = tl.arange(0, block_batch)
    partial_offsets = batch_offsets * hidden + col

    acc0 = tl.load(partial0_ptr + partial_offsets).to(tl.float32)
    acc1 = tl.load(partial1_ptr + partial_offsets).to(tl.float32)
    acc3 = tl.load(partial3_ptr + partial_offsets).to(tl.float32)

    tl.store(out0_ptr + col, tl.sum(acc0, axis=0))
    tl.store(out1_ptr + col, tl.sum(acc1, axis=0))
    tl.store(out3_ptr + col, tl.sum(acc3, axis=0).to(tl.bfloat16).to(tl.float32))


# f62c5e26: (T([32768,128], bf16), T([256,128,128], f32), ...)
@oracle_impl(hardware="B200", point="f62c5e26", xblock=16, num_warps=8, final_warps=8)
def oracle_forward(inputs, *, xblock: int, num_warps: int, final_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        _shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    full_shape = tuple(int(dim) for dim in arg1_1.shape)
    flat_shape = tuple(int(dim) for dim in shape3)
    sum_shape = tuple(int(dim) for dim in shape1)
    sum2_shape = tuple(int(dim) for dim in shape2)
    sum3_shape = tuple(int(dim) for dim in shape4)

    side_f32 = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    side_bf16 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    partial0 = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    partial1 = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    partial3 = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out0 = torch.empty_strided(sum_shape, (1,), device=arg1_1.device, dtype=torch.float32)
    out1 = torch.empty_strided(sum2_shape, (1,), device=arg1_1.device, dtype=torch.float32)
    out3 = torch.empty_strided(sum3_shape, (1,), device=arg1_1.device, dtype=torch.float32)

    xnumel = BATCH * HIDDEN
    grid = (triton.cdiv(xnumel, xblock),)
    _token_partials_and_side_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        side_f32,
        side_bf16,
        partial0,
        partial1,
        partial3,
        arg0_s0=arg0_1.stride(0),
        arg0_s1=arg0_1.stride(1),
        arg1_s0=arg1_1.stride(0),
        arg1_s1=arg1_1.stride(1),
        arg1_s2=arg1_1.stride(2),
        arg2_s0=arg2_1.stride(0),
        arg2_s1=arg2_1.stride(1),
        arg2_s2=arg2_1.stride(2),
        tokens=TOKENS,
        hidden=HIDDEN,
        xnumel=xnumel,
        xblock=xblock,
        rblock=TOKENS,
        num_warps=num_warps,
    )
    _finalize_batch_partials_kernel[(HIDDEN,)](
        partial0,
        partial1,
        partial3,
        out0,
        out1,
        out3,
        hidden=HIDDEN,
        block_batch=BATCH,
        num_warps=final_warps,
    )
    return out0, side_f32, out1, side_bf16, side_bf16.permute(1, 0), out3
