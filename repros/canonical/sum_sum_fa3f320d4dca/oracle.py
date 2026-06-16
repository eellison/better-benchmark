"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet BN-backward captured scope, including the observable contiguous clone of `arg0 + arg1`, the channels-last copy output, both high-channel per-channel reductions, the dependent scale-gradient vector, and the final bf16 channels-last input-gradient tensor; Inductor currently schedules the memory-format-changing add/clone/copy producer, the sibling `sum([0, 2, 3])` reductions, and the dependent BN-backward epilogue as generic materialized pointwise and reduction work, leaving avoidable passes over the sliced producer and limited reduction-axis parallelism; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that preserves visible mixed-layout side outputs while feeding a dependent full-tensor epilogue; the fix is COOPERATIVE_SPLIT_K: split the reduced N/H/W domain into channel partials, co-finalize the sibling summaries once, and fuse the finalized summaries into the channels-last tensor/vector epilogues."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_IN = 112
C = 56
SLICE_START = 56
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
OUT_NUMEL = N * C * HW
REDUCE_SCALE = 9.964923469387754e-06
REDUCE_R = 128
REDUCE_PARTIALS = TOTAL_SPATIAL // REDUCE_R


@triton.jit
def _add_clone_copy_kernel(
    arg0_ptr,
    arg1_ptr,
    clone_ptr,
    copy_ptr,
    YNUMEL_: tl.constexpr,
    XNUMEL_: tl.constexpr,
    C_IN_: tl.constexpr,
    HW_: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    mask = (y < YNUMEL_) & (x < XNUMEL_)

    c = y % C_IN_
    n = y // C_IN_
    channels_last_offset = c + C_IN_ * x + (C_IN_ * HW_) * n
    contiguous_offset = x + HW_ * y
    value = tl.load(arg0_ptr + channels_last_offset, mask=mask, other=0.0).to(
        tl.float32
    ) + tl.load(arg1_ptr + channels_last_offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(clone_ptr + contiguous_offset, value, mask=mask)
    tl.store(copy_ptr + channels_last_offset, value, mask=mask)


@triton.jit
def _partial_reduce_128_kernel(
    copy_ptr,
    arg2_ptr,
    mean_ptr,
    partial_prod_ptr,
    partial_sum_ptr,
    XNUMEL_: tl.constexpr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    REDUCE_R_: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    xmask = xindex < XNUMEL_
    rmask = r < REDUCE_R_
    mask = xmask & rmask

    c = xindex % C_
    partial = xindex // C_
    copy_offsets = SLICE_START_ + c + C_IN_ * r + (C_IN_ * REDUCE_R_) * partial
    arg2_offsets = c + C_ * r + (C_ * REDUCE_R_) * partial

    x = tl.load(copy_ptr + copy_offsets, mask=mask, other=0.0).to(tl.float32)
    centered = tl.load(arg2_ptr + arg2_offsets, mask=mask, other=0.0).to(
        tl.float32
    ) - tl.load(mean_ptr + c, mask=xmask, other=0.0).to(tl.float32)
    prod = x * centered
    tl.store(partial_prod_ptr + xindex, tl.sum(prod, axis=1)[:, None], mask=xmask)
    tl.store(partial_sum_ptr + xindex, tl.sum(x, axis=1)[:, None], mask=xmask)


@triton.jit
def _finalize_partials_kernel(
    partial_prod_ptr,
    partial_sum_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_out_ptr,
    vector_out_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    C_: tl.constexpr,
    PARTIALS_: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
):
    c = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    mask = (c < C_) & (r < PARTIALS_)
    offsets = c + C_ * r

    prod_values = tl.load(partial_prod_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_prod = tl.sum(prod_values, axis=1)[:, None]
    sum_x = tl.sum(sum_values, axis=1)[:, None]
    cmask = c < C_
    invstd = tl.load(invstd_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=cmask, other=0.0).to(
        tl.float32
    )

    mean_term = sum_x * REDUCE_SCALE_
    prod_coeff = (sum_prod * REDUCE_SCALE_) * (invstd * invstd)
    output_scale = invstd * affine_weight

    tl.store(sum_out_ptr + c, sum_x, mask=cmask)
    tl.store(vector_out_ptr + c, sum_prod * invstd, mask=cmask)
    tl.store(mean_term_ptr + c, mean_term, mask=cmask)
    tl.store(prod_coeff_ptr + c, prod_coeff, mask=cmask)
    tl.store(output_scale_ptr + c, output_scale, mask=cmask)


@triton.jit
def _epilogue_kernel(
    copy_ptr,
    arg2_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    OUT_NUMEL_: tl.constexpr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    c = linear % C_
    nhw = linear // C_
    hw = nhw % HW_
    n = nhw // HW_
    wide_offsets = n * (C_IN_ * HW_) + hw * C_IN_ + SLICE_START_ + c

    x = tl.load(copy_ptr + wide_offsets).to(tl.float32)
    centered = (
        tl.load(arg2_ptr + linear).to(tl.float32)
        - tl.load(mean_ptr + c).to(tl.float32)
    )
    out = (
        (
            x
            - centered * tl.load(prod_coeff_ptr + c).to(tl.float32)
            - tl.load(mean_term_ptr + c).to(tl.float32)
        )
        * tl.load(output_scale_ptr + c).to(tl.float32)
    )
    tl.store(out_ptr + linear, out)


@oracle_impl(hardware="B200", point="036a9f6f")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs

    clone = torch.empty_strided(
        (N, C_IN, H, W),
        (C_IN * HW, HW, W, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    copy = torch.empty_strided(
        (N, C_IN, H, W),
        (C_IN * HW, 1, W * C_IN, C_IN),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_1 = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    mul_8 = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    add_yblock = 16
    add_xblock = 256
    _add_clone_copy_kernel[
        (triton.cdiv(HW, add_xblock), triton.cdiv(N * C_IN, add_yblock))
    ](
        arg0_1,
        arg1_1,
        clone,
        copy,
        YNUMEL_=N * C_IN,
        XNUMEL_=HW,
        C_IN_=C_IN,
        HW_=HW,
        YBLOCK=add_yblock,
        XBLOCK=add_xblock,
        num_warps=8,
    )

    partial_prod = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=arg0_1.device, dtype=torch.float32
    )
    partial_sum = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=arg0_1.device, dtype=torch.float32
    )
    partial_xblock = 32
    _partial_reduce_128_kernel[(triton.cdiv(C * REDUCE_PARTIALS, partial_xblock),)](
        copy,
        arg2_1,
        arg3_1,
        partial_prod,
        partial_sum,
        XNUMEL_=C * REDUCE_PARTIALS,
        C_IN_=C_IN,
        C_=C,
        SLICE_START_=SLICE_START,
        REDUCE_R_=REDUCE_R,
        XBLOCK=partial_xblock,
        RBLOCK=REDUCE_R,
        num_warps=8,
    )

    mean_term = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    prod_coeff = torch.empty_like(mean_term)
    output_scale = torch.empty_like(mean_term)
    finalize_xblock = 16
    _finalize_partials_kernel[(triton.cdiv(C, finalize_xblock),)](
        partial_prod,
        partial_sum,
        arg4_1,
        arg5_1,
        sum_1,
        mul_8,
        mean_term,
        prod_coeff,
        output_scale,
        C_=C,
        PARTIALS_=REDUCE_PARTIALS,
        XBLOCK=finalize_xblock,
        RBLOCK=triton.next_power_of_2(REDUCE_PARTIALS),
        REDUCE_SCALE_=REDUCE_SCALE,
        num_warps=8,
    )

    epilogue_block = 1024
    _epilogue_kernel[(triton.cdiv(OUT_NUMEL, epilogue_block),)](
        copy,
        arg2_1,
        arg3_1,
        mean_term,
        prod_coeff,
        output_scale,
        out,
        OUT_NUMEL_=OUT_NUMEL,
        C_IN_=C_IN,
        C_=C,
        SLICE_START_=SLICE_START,
        HW_=HW,
        BLOCK=epilogue_block,
        num_warps=4,
    )

    return clone, copy, sum_1, mul_8, out
