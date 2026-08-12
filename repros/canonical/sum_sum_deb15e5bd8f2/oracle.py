"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 BN-backward captured scope, including the observable contiguous clone of `arg0 + arg1`, the channels-last copy output, both high-channel per-channel reductions, the dependent scale-gradient vector, and the final bf16 channels-last input-gradient tensor; Inductor currently schedules the memory-format-changing add/clone/copy producer, the sibling `sum([0, 2, 3])` reductions, and the dependent BN-backward epilogue as generic materialized pointwise and reduction work over the huge `[512,16,112,112]` producer; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that preserves visible mixed-layout side outputs while feeding a dependent full-tensor epilogue; the fix is COOPERATIVE_SPLIT_K: split the reduced N/H/W domain into channel partials, co-finalize sibling summaries once, and fuse finalized summaries into the channels-last tensor/vector epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_IN = 16
C = 8
SLICE_START = 8
H = 112
W = 112
HW = H * W
OUT_NUMEL = N * C * HW
REDUCE_SCALE = 1.5570192920918366e-07
BATCHES_PER_CHUNK = 1
BATCH_CHUNKS = N // BATCHES_PER_CHUNK
CHUNK_K = BATCHES_PER_CHUNK * HW
REDUCE_R = 1024
REDUCE_PARTIALS = (N * HW) // REDUCE_R


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
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
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
    value = _add_rn_f32(
        tl.load(arg0_ptr + channels_last_offset, mask=mask, other=0.0).to(tl.float32),
        tl.load(arg1_ptr + channels_last_offset, mask=mask, other=0.0).to(tl.float32),
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(clone_ptr + contiguous_offset, value, mask=mask)
    tl.store(copy_ptr + channels_last_offset, value, mask=mask)


@triton.jit
def _partial_reduce_kernel(
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
    centered = _sub_rn_f32(
        tl.load(arg2_ptr + arg2_offsets, mask=mask, other=0.0).to(tl.float32),
        tl.load(mean_ptr + c, mask=xmask, other=0.0).to(tl.float32),
    )
    prod = _mul_rn_f32(x, centered)
    tl.store(partial_prod_ptr + xindex, tl.sum(prod, axis=1)[:, None], mask=xmask)
    tl.store(partial_sum_ptr + xindex, tl.sum(x, axis=1)[:, None], mask=xmask)


@triton.jit
def _high_add_copy_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean_ptr,
    clone_ptr,
    copy_ptr,
    partial_sum_ptr,
    partial_prod_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    CHUNK_K_: tl.constexpr,
    BATCHES_PER_CHUNK_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunk = tl.program_id(1)
    k_offsets = tl.arange(0, BLOCK_K)
    c_mask = c < C_

    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    acc_sum = tl.full((BLOCK_C,), 0.0, tl.float32)
    acc_prod = tl.full((BLOCK_C,), 0.0, tl.float32)

    for base in tl.range(0, CHUNK_K_, BLOCK_K):
        k = base + k_offsets
        k_mask = k < CHUNK_K_
        batch_in_chunk = k // HW_
        hw = k - batch_in_chunk * HW_
        n = chunk * BATCHES_PER_CHUNK_ + batch_in_chunk

        wide_offsets = (
            n[:, None] * (C_IN_ * HW_)
            + hw[:, None] * C_IN_
            + (SLICE_START_ + c[None, :])
        )
        clone_offsets = (
            n[:, None] * (C_IN_ * HW_)
            + (SLICE_START_ + c[None, :]) * HW_
            + hw[:, None]
        )
        narrow_offsets = n[:, None] * (C_ * HW_) + hw[:, None] * C_ + c[None, :]
        mask = k_mask[:, None] & c_mask[None, :]

        x = _add_rn_f32(
            tl.load(arg0_ptr + wide_offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(arg1_ptr + wide_offsets, mask=mask, other=0.0).to(tl.float32),
        )
        x_bf16 = x.to(tl.bfloat16, fp_downcast_rounding="rtne")
        x_f32 = x_bf16.to(tl.float32)
        tl.store(copy_ptr + wide_offsets, x_bf16, mask=mask)
        tl.store(clone_ptr + clone_offsets, x_bf16, mask=mask)

        centered = _sub_rn_f32(
            tl.load(arg2_ptr + narrow_offsets, mask=mask, other=0.0).to(tl.float32),
            mean[None, :],
        )
        acc_sum += tl.sum(x_f32, axis=0)
        acc_prod += tl.sum(_mul_rn_f32(x_f32, centered), axis=0)

    partial_offsets = chunk * C_ + c
    tl.store(partial_sum_ptr + partial_offsets, acc_sum, mask=c_mask)
    tl.store(partial_prod_ptr + partial_offsets, acc_prod, mask=c_mask)


@triton.jit
def _low_add_copy_kernel(
    arg0_ptr,
    arg1_ptr,
    clone_ptr,
    copy_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    CHUNK_K_: tl.constexpr,
    BATCHES_PER_CHUNK_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunk = tl.program_id(1)
    k_offsets = tl.arange(0, BLOCK_K)
    c_mask = c < C_

    for base in tl.range(0, CHUNK_K_, BLOCK_K):
        k = base + k_offsets
        k_mask = k < CHUNK_K_
        batch_in_chunk = k // HW_
        hw = k - batch_in_chunk * HW_
        n = chunk * BATCHES_PER_CHUNK_ + batch_in_chunk

        copy_offsets = n[:, None] * (C_IN_ * HW_) + hw[:, None] * C_IN_ + c[None, :]
        clone_offsets = n[:, None] * (C_IN_ * HW_) + c[None, :] * HW_ + hw[:, None]
        mask = k_mask[:, None] & c_mask[None, :]

        value = _add_rn_f32(
            tl.load(arg0_ptr + copy_offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(arg1_ptr + copy_offsets, mask=mask, other=0.0).to(tl.float32),
        ).to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(copy_ptr + copy_offsets, value, mask=mask)
        tl.store(clone_ptr + clone_offsets, value, mask=mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_prod_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_out_ptr,
    vector_out_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    C_: tl.constexpr,
    BATCH_CHUNKS_: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
):
    c = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < BATCH_CHUNKS_
    offsets = chunks * C_ + c

    sum_values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    prod_values = tl.load(partial_prod_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(sum_values, axis=0)
    sum_prod = tl.sum(prod_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)

    mean_term = _mul_rn_f32(sum_x, REDUCE_SCALE_)
    prod_scaled = _mul_rn_f32(sum_prod, REDUCE_SCALE_)
    invstd_sq = _mul_rn_f32(invstd, invstd)
    prod_coeff = _mul_rn_f32(prod_scaled, invstd_sq)
    output_scale = _mul_rn_f32(invstd, affine_weight)

    tl.store(sum_out_ptr + c, sum_x)
    tl.store(vector_out_ptr + c, _mul_rn_f32(sum_prod, invstd))
    tl.store(mean_term_ptr + c, mean_term)
    tl.store(prod_coeff_ptr + c, prod_coeff)
    tl.store(output_scale_ptr + c, output_scale)


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
    mask = linear < OUT_NUMEL_

    c = linear % C_
    nhw = linear // C_
    hw = nhw % HW_
    n = nhw // HW_
    wide_offsets = n * (C_IN_ * HW_) + hw * C_IN_ + SLICE_START_ + c

    x = tl.load(copy_ptr + wide_offsets, mask=mask, other=0.0).to(tl.float32)
    centered = _sub_rn_f32(
        tl.load(arg2_ptr + linear, mask=mask, other=0.0).to(tl.float32),
        tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32),
    )
    correction = _mul_rn_f32(centered, tl.load(prod_coeff_ptr + c, mask=mask, other=0.0).to(tl.float32))
    residual = _sub_rn_f32(_sub_rn_f32(x, correction), tl.load(mean_term_ptr + c, mask=mask, other=0.0).to(tl.float32))
    out = _mul_rn_f32(residual, tl.load(output_scale_ptr + c, mask=mask, other=0.0).to(tl.float32))
    tl.store(out_ptr + linear, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# (T([512,16,112,112], bf16, channels-last), T([512,16,112,112], bf16, channels-last), T([512,8,112,112], bf16, channels-last), T([1,8,1,1], f32), T([8], f32), T([8], f32))
@oracle_impl(
    hardware="B200",
    point="a8508031",
    BLOCK_C=8,
    ADD_YBLOCK=16,
    ADD_XBLOCK=256,
    PARTIAL_XBLOCK=4,
    PARTIAL_RBLOCK=1024,
    EPILOGUE_BLOCK=1024,
    num_warps_add=8,
    num_warps_reduce=8,
    num_warps_final=8,
    num_warps_epilogue=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    ADD_YBLOCK: int,
    ADD_XBLOCK: int,
    PARTIAL_XBLOCK: int,
    PARTIAL_RBLOCK: int,
    EPILOGUE_BLOCK: int,
    num_warps_add: int,
    num_warps_reduce: int,
    num_warps_final: int,
    num_warps_epilogue: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    clone = torch.empty_strided(
        (N, C_IN, H, W),
        (C_IN * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    copy = torch.empty_strided(
        (N, C_IN, H, W),
        (C_IN * HW, 1, W * C_IN, C_IN),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty_strided(
        (REDUCE_PARTIALS, C),
        (C, 1),
        device=device,
        dtype=torch.float32,
    )
    partial_prod = torch.empty_strided(
        (REDUCE_PARTIALS, C),
        (C, 1),
        device=device,
        dtype=torch.float32,
    )

    _add_clone_copy_kernel[
        (triton.cdiv(HW, ADD_XBLOCK), triton.cdiv(N * C_IN, ADD_YBLOCK))
    ](
        arg0_1,
        arg1_1,
        clone,
        copy,
        YNUMEL_=N * C_IN,
        XNUMEL_=HW,
        C_IN_=C_IN,
        HW_=HW,
        YBLOCK=ADD_YBLOCK,
        XBLOCK=ADD_XBLOCK,
        num_warps=num_warps_add,
        num_stages=3,
    )
    _partial_reduce_kernel[(triton.cdiv(C * REDUCE_PARTIALS, PARTIAL_XBLOCK),)](
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
        XBLOCK=PARTIAL_XBLOCK,
        RBLOCK=PARTIAL_RBLOCK,
        num_warps=num_warps_reduce,
        num_stages=3,
    )

    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    mul_8 = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_kernel[(C,)](
        partial_sum,
        partial_prod,
        arg4_1,
        arg5_1,
        sum_1,
        mul_8,
        mean_term,
        prod_coeff,
        output_scale,
        C_=C,
        BATCH_CHUNKS_=REDUCE_PARTIALS,
        BLOCK_CHUNKS=_next_power_of_2(REDUCE_PARTIALS),
        REDUCE_SCALE_=REDUCE_SCALE,
        num_warps=num_warps_final,
        num_stages=3,
    )

    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    _epilogue_kernel[(triton.cdiv(OUT_NUMEL, EPILOGUE_BLOCK),)](
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
        BLOCK=EPILOGUE_BLOCK,
        num_warps=num_warps_epilogue,
        num_stages=3,
    )

    return clone, copy, sum_1, mul_8, out
