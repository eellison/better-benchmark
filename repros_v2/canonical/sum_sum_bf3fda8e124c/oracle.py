"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 BN-backward captured scope, including the visible contiguous clone of `arg0 + arg1`, the channels-last copy output, both per-channel f32 reductions over the high-channel slice, the dependent scale-gradient vector, and the final bf16 channels-last input-gradient tensor, whereas Inductor schedules the mixed-layout add/clone/copy producer, sibling `sum([0, 2, 3])` reductions, and reduction-dependent BN-backward epilogue as separate generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that preserves visible mixed-layout side outputs while feeding a dependent full-tensor epilogue with exact bf16 cast boundaries; the fix is COOPERATIVE_SPLIT_K: split the reduced N/H/W domain into channel partials, co-finalize the sibling summaries once, and fuse the finalized summaries into the channels-last tensor/vector epilogues."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_IN = 24
C = 12
SLICE_START = 12
H = 56
W = 56
HW = H * W
TOTAL_SPATIAL = N * HW
OUT_NUMEL = N * C * HW
REDUCE_SCALE = 6.228077168367346e-07
REDUCE_R = 1024
REDUCE_PARTIALS = TOTAL_SPATIAL // REDUCE_R


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _add_clone_copy_kernel(
    arg0_ptr,
    arg1_ptr,
    clone_ptr,
    copy_ptr,
    YNUMEL: tl.constexpr,
    XNUMEL: tl.constexpr,
    C_IN_N: tl.constexpr,
    HW_N: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    mask = (y < YNUMEL) & (x < XNUMEL)

    c = y % C_IN_N
    n = y // C_IN_N
    channels_last_offset = c + C_IN_N * x + (C_IN_N * HW_N) * n
    contiguous_offset = x + HW_N * y

    lhs = tl.load(arg0_ptr + channels_last_offset, mask=mask, other=0.0)
    rhs = tl.load(arg1_ptr + channels_last_offset, mask=mask, other=0.0)
    value = _bf16_add(lhs, rhs)
    tl.store(clone_ptr + contiguous_offset, value, mask=mask)
    tl.store(copy_ptr + channels_last_offset, value, mask=mask)


@triton.jit
def _partial_reduce_kernel(
    copy_ptr,
    activation_ptr,
    mean_ptr,
    partial_prod_ptr,
    partial_sum_ptr,
    XNUMEL: tl.constexpr,
    C_IN_N: tl.constexpr,
    C_N: tl.constexpr,
    SLICE_START_N: tl.constexpr,
    REDUCE_R_N: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    xmask = xindex < XNUMEL
    rmask = r < REDUCE_R_N
    mask = xmask & rmask

    c = xindex % C_N
    partial = xindex // C_N
    copy_offsets = SLICE_START_N + c + C_IN_N * r + (C_IN_N * REDUCE_R_N) * partial
    activation_offsets = c + C_N * r + (C_N * REDUCE_R_N) * partial

    x = tl.load(copy_ptr + copy_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + activation_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=xmask, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean)
    prod = _f32_mul(x, centered)
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
    C_N: tl.constexpr,
    PARTIALS_N: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    REDUCE_SCALE_N: tl.constexpr,
):
    c = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    mask = (c < C_N) & (r < PARTIALS_N)
    offsets = c + C_N * r

    prod_values = tl.load(partial_prod_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_prod = tl.sum(prod_values, axis=1)[:, None]
    sum_x = tl.sum(sum_values, axis=1)[:, None]
    cmask = c < C_N
    invstd = tl.load(invstd_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=cmask, other=0.0).to(
        tl.float32
    )

    mean_term = _f32_mul(sum_x, REDUCE_SCALE_N)
    prod_coeff = _f32_mul(_f32_mul(sum_prod, REDUCE_SCALE_N), _f32_mul(invstd, invstd))
    output_scale = _f32_mul(invstd, affine_weight)

    tl.store(sum_out_ptr + c, sum_x, mask=cmask)
    tl.store(vector_out_ptr + c, _f32_mul(sum_prod, invstd), mask=cmask)
    tl.store(mean_term_ptr + c, mean_term, mask=cmask)
    tl.store(prod_coeff_ptr + c, prod_coeff, mask=cmask)
    tl.store(output_scale_ptr + c, output_scale, mask=cmask)


@triton.jit
def _epilogue_kernel(
    copy_ptr,
    activation_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    OUT_NUMEL_N: tl.constexpr,
    C_IN_N: tl.constexpr,
    C_N: tl.constexpr,
    SLICE_START_N: tl.constexpr,
    HW_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < OUT_NUMEL_N

    c = linear % C_N
    nhw = linear // C_N
    hw = nhw % HW_N
    n = nhw // HW_N
    wide_offsets = n * (C_IN_N * HW_N) + hw * C_IN_N + SLICE_START_N + c

    x = tl.load(copy_ptr + wide_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + linear, mask=mask, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32))
    out = _f32_mul(
        _f32_sub(
            _f32_sub(x, _f32_mul(centered, tl.load(prod_coeff_ptr + c, mask=mask, other=0.0).to(tl.float32))),
            tl.load(mean_term_ptr + c, mask=mask, other=0.0).to(tl.float32),
        ),
        tl.load(output_scale_ptr + c, mask=mask, other=0.0).to(tl.float32),
    )
    tl.store(out_ptr + linear, out, mask=mask)


@oracle_impl(
    hardware="B200",
    point="a63c41b1",
    ADD_YBLOCK=16,
    ADD_XBLOCK=256,
    PARTIAL_XBLOCK=16,
    FINAL_XBLOCK=8,
    EPILOGUE_BLOCK=512,
    add_warps=8,
    partial_warps=8,
    final_warps=8,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    ADD_YBLOCK: int,
    ADD_XBLOCK: int,
    PARTIAL_XBLOCK: int,
    FINAL_XBLOCK: int,
    EPILOGUE_BLOCK: int,
    add_warps: int,
    partial_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
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

    _add_clone_copy_kernel[
        (triton.cdiv(HW, ADD_XBLOCK), triton.cdiv(N * C_IN, ADD_YBLOCK))
    ](
        arg0_1,
        arg1_1,
        clone,
        copy,
        YNUMEL=N * C_IN,
        XNUMEL=HW,
        C_IN_N=C_IN,
        HW_N=HW,
        YBLOCK=ADD_YBLOCK,
        XBLOCK=ADD_XBLOCK,
        num_warps=add_warps,
    )

    partial_prod = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=arg0_1.device, dtype=torch.float32
    )
    partial_sum = torch.empty_strided(
        (C, REDUCE_PARTIALS), (1, C), device=arg0_1.device, dtype=torch.float32
    )
    _partial_reduce_kernel[(triton.cdiv(C * REDUCE_PARTIALS, PARTIAL_XBLOCK),)](
        copy,
        arg2_1,
        arg3_1,
        partial_prod,
        partial_sum,
        XNUMEL=C * REDUCE_PARTIALS,
        C_IN_N=C_IN,
        C_N=C,
        SLICE_START_N=SLICE_START,
        REDUCE_R_N=REDUCE_R,
        XBLOCK=PARTIAL_XBLOCK,
        RBLOCK=REDUCE_R,
        num_warps=partial_warps,
    )

    mean_term = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    prod_coeff = torch.empty_like(mean_term)
    output_scale = torch.empty_like(mean_term)
    _finalize_partials_kernel[(triton.cdiv(C, FINAL_XBLOCK),)](
        partial_prod,
        partial_sum,
        arg4_1,
        arg5_1,
        sum_1,
        mul_8,
        mean_term,
        prod_coeff,
        output_scale,
        C_N=C,
        PARTIALS_N=REDUCE_PARTIALS,
        XBLOCK=FINAL_XBLOCK,
        RBLOCK=triton.next_power_of_2(REDUCE_PARTIALS),
        REDUCE_SCALE_N=REDUCE_SCALE,
        num_warps=final_warps,
    )

    _epilogue_kernel[(triton.cdiv(OUT_NUMEL, EPILOGUE_BLOCK),)](
        copy,
        arg2_1,
        arg3_1,
        mean_term,
        prod_coeff,
        output_scale,
        out,
        OUT_NUMEL_N=OUT_NUMEL,
        C_IN_N=C_IN,
        C_N=C,
        SLICE_START_N=SLICE_START,
        HW_N=HW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
    )

    return clone, copy, sum_1, mul_8, out
