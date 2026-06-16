"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileBERT bf16 multi-output reduction scope by token-tiling the shared `bf16 + bf16 -> fp32` producer, writing the required bf16-scaled side tensor once, returning its transpose as a metadata alias, and finalizing all three `[128]` column reductions from common batch partials, whereas Inductor schedules the shared add, sibling product reduction, bf16 side materialization, transpose view, and bf16-rounded side-output reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that preserves explicit bf16/fp32 conversion boundaries while coordinating layout-producing side outputs and compatible column partials from one producer; the fix is COOPERATIVE_SPLIT_K: add a row-token multi-output reduction plan that shares the producer, sinks view-only layout algebra into output planning, and finalizes all compatible column reductions together."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128


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
    lhs_ptr,
    rhs_ptr,
    product_rhs_ptr,
    scale_ptr,
    side_bf16_ptr,
    partial0_ptr,
    partial1_ptr,
    partial2_ptr,
    hidden: tl.constexpr,
    xblock: tl.constexpr,
    rblock: tl.constexpr,
):
    xindex = tl.program_id(0) * xblock + tl.arange(0, xblock)[:, None]
    token_offsets = tl.arange(0, rblock)[None, :]

    hidden_index = xindex % hidden
    batch_index = xindex // hidden
    flat_row = batch_index * rblock + token_offsets
    offsets = flat_row * hidden + hidden_index

    x = _add_rn(
        tl.load(lhs_ptr + offsets).to(tl.float32),
        tl.load(rhs_ptr + offsets).to(tl.float32),
    )
    product_rhs = tl.load(product_rhs_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + hidden_index).to(tl.float32)
    side_bf16 = _mul_rn(x, scale).to(tl.bfloat16)

    tl.store(side_bf16_ptr + offsets, side_bf16)

    partial_offsets = batch_index * hidden + hidden_index
    tl.store(partial0_ptr + partial_offsets, tl.sum(x, axis=1)[:, None])
    tl.store(
        partial1_ptr + partial_offsets,
        tl.sum(_mul_rn(x, product_rhs), axis=1)[:, None],
    )
    tl.store(
        partial2_ptr + partial_offsets,
        tl.sum(side_bf16.to(tl.float32), axis=1)[:, None],
    )


@triton.jit
def _finalize_batch_partials_kernel(
    partial0_ptr,
    partial1_ptr,
    partial2_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    hidden: tl.constexpr,
    block_batch: tl.constexpr,
):
    col = tl.program_id(0)
    batch_offsets = tl.arange(0, block_batch)
    offsets = batch_offsets * hidden + col

    acc0 = tl.load(partial0_ptr + offsets).to(tl.float32)
    acc1 = tl.load(partial1_ptr + offsets).to(tl.float32)
    acc2 = tl.load(partial2_ptr + offsets).to(tl.float32)

    tl.store(out0_ptr + col, tl.sum(acc0, axis=0))
    tl.store(out1_ptr + col, tl.sum(acc1, axis=0))
    tl.store(out2_ptr + col, tl.sum(acc2, axis=0).to(tl.bfloat16).to(tl.float32))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 02a109be: (T([32768,128], bf16) x3, T([128], f32), ...)
@oracle_impl(
    hardware="B200",
    point="02a109be",
    xblock=32,
    num_warps=4,
    final_warps=8,
)
def oracle_forward(inputs, *, xblock: int, num_warps: int, final_warps: int):
    (
        lhs,
        rhs,
        product_rhs,
        scale,
        _shape0,
        _shape1,
        sum0_shape_param,
        _shape3,
        sum1_shape_param,
        flat_shape_param,
        sum2_shape_param,
    ) = inputs

    hidden = int(lhs.shape[1])
    flat_shape = _shape_tuple(flat_shape_param)
    sum0_shape = _shape_tuple(sum0_shape_param)
    sum1_shape = _shape_tuple(sum1_shape_param)
    sum2_shape = _shape_tuple(sum2_shape_param)

    side_bf16 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=lhs.device,
        dtype=torch.bfloat16,
    )
    partial0 = torch.empty_strided(
        (BATCH, hidden),
        (hidden, 1),
        device=lhs.device,
        dtype=torch.float32,
    )
    partial1 = torch.empty_strided(
        (BATCH, hidden),
        (hidden, 1),
        device=lhs.device,
        dtype=torch.float32,
    )
    partial2 = torch.empty_strided(
        (BATCH, hidden),
        (hidden, 1),
        device=lhs.device,
        dtype=torch.float32,
    )
    out0 = torch.empty_strided(sum0_shape, (1,), device=lhs.device, dtype=torch.float32)
    out1 = torch.empty_strided(sum1_shape, (1,), device=lhs.device, dtype=torch.float32)
    out2 = torch.empty_strided(sum2_shape, (1,), device=lhs.device, dtype=torch.float32)

    xnumel = BATCH * hidden
    _token_partials_and_side_kernel[(triton.cdiv(xnumel, xblock),)](
        lhs,
        rhs,
        product_rhs,
        scale,
        side_bf16,
        partial0,
        partial1,
        partial2,
        hidden=hidden,
        xblock=xblock,
        rblock=TOKENS,
        num_warps=num_warps,
    )
    _finalize_batch_partials_kernel[(hidden,)](
        partial0,
        partial1,
        partial2,
        out0,
        out1,
        out2,
        hidden=hidden,
        block_batch=BATCH,
        num_warps=final_warps,
    )

    return out0, out1, side_bf16, side_bf16.permute(1, 0), out2
