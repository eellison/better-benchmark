"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet dual batch-norm-backward return tuple by reducing the shared bf16 add/copy/clone producer and its high-channel 40-channel slice in one split-K Triton reduction, emitting the returned channels-last add copy, contiguous full-gradient tensor, channels-last slice-gradient tensor, and both scale-gradient vectors while preserving the captured bf16 producer and output cast boundaries, whereas Inductor schedules the memory-format copy/clone/slice path, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogues as separate generic pointwise/reduction regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output lowering that shares overlapping channel partials across layout-changing producers and dependent side outputs; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile the `(N, H, W)` reduction domain for compatible channel reductions, accumulate the full and sliced branch summaries together, and fuse the downstream tensor/vector epilogues with layout-aware stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 512
CHANNELS_FULL = 80
CHANNELS_SLICE = 40
SLICE_START = 40
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
REDUCTION_K = BATCH * HW
TOTAL_FULL = BATCH * CHANNELS_FULL * HW
TOTAL_SLICE = BATCH * CHANNELS_SLICE * HW
NORM_SCALE = 9.964923469387754e-06
REDUCE_BLOCK_K = 1024
FINAL_BLOCK_K = 128
POINTWISE_BLOCK = 256


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
def _partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    rhs_full_ptr,
    mean_full_ptr,
    rhs_slice_ptr,
    mean_slice_ptr,
    out_copy_ptr,
    partial_x_full_ptr,
    partial_x_rhs_full_ptr,
    partial_x_slice_ptr,
    partial_x_rhs_slice_ptr,
    NBLOCKS: tl.constexpr,
    REDUCTION_K_: tl.constexpr,
    CHANNELS_FULL_: tl.constexpr,
    CHANNELS_SLICE_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    WIDTH_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    block_id = tl.program_id(1)
    k = block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < REDUCTION_K_
    batch = k // HW_
    spatial = k - batch * HW_
    h = spatial // WIDTH_
    w = spatial - h * WIDTH_

    dense_offset = batch * (CHANNELS_FULL_ * HW_) + channel * HW_ + spatial
    channels_last_offset = batch * (CHANNELS_FULL_ * HW_) + spatial * CHANNELS_FULL_ + channel

    x_f32 = _add_rn(
        tl.load(arg0_ptr + dense_offset, mask=mask, other=0.0).to(tl.float32),
        tl.load(arg1_ptr + channels_last_offset, mask=mask, other=0.0).to(tl.float32),
    )
    x_bf16 = x_f32.to(tl.bfloat16)
    tl.store(out_copy_ptr + channels_last_offset, x_bf16, mask=mask)
    x = x_bf16.to(tl.float32)

    centered_full = _sub_rn(
        tl.load(rhs_full_ptr + channels_last_offset, mask=mask, other=0.0).to(tl.float32),
        tl.load(mean_full_ptr + channel).to(tl.float32),
    )
    partial_full_offset = channel * NBLOCKS + block_id
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    tl.store(partial_x_full_ptr + partial_full_offset, sum_x)
    tl.store(
        partial_x_rhs_full_ptr + partial_full_offset,
        tl.sum(tl.where(mask, _mul_rn(x, centered_full), 0.0), axis=0),
    )

    channel_slice = channel - SLICE_START_
    in_slice = channel >= SLICE_START_
    slice_offset = batch * (CHANNELS_SLICE_ * HW_) + spatial * CHANNELS_SLICE_ + channel_slice
    mask_slice = mask & in_slice
    centered_slice = _sub_rn(
        tl.load(rhs_slice_ptr + slice_offset, mask=mask_slice, other=0.0).to(tl.float32),
        tl.load(mean_slice_ptr + channel_slice, mask=in_slice, other=0.0).to(tl.float32),
    )
    partial_slice_offset = channel_slice * NBLOCKS + block_id
    tl.store(partial_x_slice_ptr + partial_slice_offset, sum_x, mask=in_slice)
    tl.store(
        partial_x_rhs_slice_ptr + partial_slice_offset,
        tl.sum(tl.where(mask_slice, _mul_rn(x, centered_slice), 0.0), axis=0),
        mask=in_slice,
    )


@triton.jit
def _finalize_kernel(
    partial_x_ptr,
    partial_x_rhs_ptr,
    rsqrt_ptr,
    weight_ptr,
    mean_term_ptr,
    coeff_ptr,
    fused_weight_ptr,
    out_x_ptr,
    out_scale_ptr,
    NBLOCKS: tl.constexpr,
    NORM_SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < NBLOCKS
    partial_offsets = channel * NBLOCKS + offsets
    sum_x = tl.sum(
        tl.load(partial_x_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_x_rhs = tl.sum(
        tl.load(partial_x_rhs_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    rsqrt = tl.load(rsqrt_ptr + channel).to(tl.float32)
    weight = tl.load(weight_ptr + channel).to(tl.float32)
    tl.store(mean_term_ptr + channel, _mul_rn(sum_x, NORM_SCALE_))
    tl.store(coeff_ptr + channel, _mul_rn(_mul_rn(sum_x_rhs, NORM_SCALE_), _mul_rn(rsqrt, rsqrt)))
    tl.store(fused_weight_ptr + channel, _mul_rn(rsqrt, weight))
    tl.store(out_x_ptr + channel, sum_x)
    tl.store(out_scale_ptr + channel, _mul_rn(sum_x_rhs, rsqrt))


@triton.jit
def _pointwise_full_kernel(
    out_copy_ptr,
    rhs_ptr,
    mean_ptr,
    mean_term_ptr,
    coeff_ptr,
    fused_weight_ptr,
    out_ptr,
    TOTAL_FULL_: tl.constexpr,
    CHANNELS_FULL_: tl.constexpr,
    HW_: tl.constexpr,
    WIDTH_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL_FULL_
    channel = (offsets // HW_) % CHANNELS_FULL_
    spatial = offsets % HW_
    batch = offsets // (CHANNELS_FULL_ * HW_)
    cl_offset = batch * (CHANNELS_FULL_ * HW_) + spatial * CHANNELS_FULL_ + channel
    x = tl.load(out_copy_ptr + cl_offset, mask=mask, other=0.0).to(tl.float32)
    centered = _sub_rn(
        tl.load(rhs_ptr + cl_offset, mask=mask, other=0.0).to(tl.float32),
        tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32),
    )
    value = _sub_rn(
        _sub_rn(
            x,
            _mul_rn(centered, tl.load(coeff_ptr + channel, mask=mask, other=0.0).to(tl.float32)),
        ),
        tl.load(mean_term_ptr + channel, mask=mask, other=0.0).to(tl.float32),
    )
    value = _mul_rn(value, tl.load(fused_weight_ptr + channel, mask=mask, other=0.0).to(tl.float32))
    tl.store(out_ptr + offsets, value.to(tl.bfloat16), mask=mask)


@triton.jit
def _pointwise_slice_kernel(
    out_copy_ptr,
    rhs_ptr,
    mean_ptr,
    mean_term_ptr,
    coeff_ptr,
    fused_weight_ptr,
    out_ptr,
    TOTAL_SLICE_: tl.constexpr,
    CHANNELS_FULL_: tl.constexpr,
    CHANNELS_SLICE_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL_SLICE_
    channel = offsets % CHANNELS_SLICE_
    tmp = offsets // CHANNELS_SLICE_
    spatial = tmp % HW_
    batch = tmp // HW_
    source_offset = batch * (CHANNELS_FULL_ * HW_) + spatial * CHANNELS_FULL_ + channel + SLICE_START_
    rhs_offset = batch * (CHANNELS_SLICE_ * HW_) + spatial * CHANNELS_SLICE_ + channel
    x = tl.load(out_copy_ptr + source_offset, mask=mask, other=0.0).to(tl.float32)
    centered = _sub_rn(
        tl.load(rhs_ptr + rhs_offset, mask=mask, other=0.0).to(tl.float32),
        tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32),
    )
    value = _sub_rn(
        _sub_rn(
            x,
            _mul_rn(centered, tl.load(coeff_ptr + channel, mask=mask, other=0.0).to(tl.float32)),
        ),
        tl.load(mean_term_ptr + channel, mask=mask, other=0.0).to(tl.float32),
    )
    value = _mul_rn(value, tl.load(fused_weight_ptr + channel, mask=mask, other=0.0).to(tl.float32))
    tl.store(out_ptr + offsets, value.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="b55d777f")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    nblocks = triton.cdiv(REDUCTION_K, REDUCE_BLOCK_K)

    out_copy = torch.empty_strided(
        (BATCH, CHANNELS_FULL, HEIGHT, WIDTH),
        (CHANNELS_FULL * HW, 1, WIDTH * CHANNELS_FULL, CHANNELS_FULL),
        device=device,
        dtype=torch.bfloat16,
    )
    out_full = torch.empty_strided(
        (BATCH, CHANNELS_FULL, HEIGHT, WIDTH),
        (CHANNELS_FULL * HW, HW, WIDTH, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_slice = torch.empty_strided(
        (BATCH, CHANNELS_SLICE, HEIGHT, WIDTH),
        (CHANNELS_SLICE * HW, 1, WIDTH * CHANNELS_SLICE, CHANNELS_SLICE),
        device=device,
        dtype=torch.bfloat16,
    )

    partial_x_full = torch.empty((CHANNELS_FULL, nblocks), device=device, dtype=torch.float32)
    partial_x_rhs_full = torch.empty_like(partial_x_full)
    partial_x_slice = torch.empty((CHANNELS_SLICE, nblocks), device=device, dtype=torch.float32)
    partial_x_rhs_slice = torch.empty_like(partial_x_slice)

    mean_term_full = torch.empty((CHANNELS_FULL,), device=device, dtype=torch.float32)
    coeff_full = torch.empty_like(mean_term_full)
    fused_weight_full = torch.empty_like(mean_term_full)
    out_x_full = torch.empty_like(mean_term_full)
    out_scale_full = torch.empty_like(mean_term_full)

    mean_term_slice = torch.empty((CHANNELS_SLICE,), device=device, dtype=torch.float32)
    coeff_slice = torch.empty_like(mean_term_slice)
    fused_weight_slice = torch.empty_like(mean_term_slice)
    out_x_slice = torch.empty_like(mean_term_slice)
    out_scale_slice = torch.empty_like(mean_term_slice)

    _partial_reduce_kernel[(CHANNELS_FULL, nblocks)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg6_1,
        arg7_1,
        out_copy,
        partial_x_full,
        partial_x_rhs_full,
        partial_x_slice,
        partial_x_rhs_slice,
        NBLOCKS=nblocks,
        REDUCTION_K_=REDUCTION_K,
        CHANNELS_FULL_=CHANNELS_FULL,
        CHANNELS_SLICE_=CHANNELS_SLICE,
        SLICE_START_=SLICE_START,
        HW_=HW,
        WIDTH_=WIDTH,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=4,
    )
    _finalize_kernel[(CHANNELS_FULL,)](
        partial_x_full,
        partial_x_rhs_full,
        arg4_1,
        arg5_1,
        mean_term_full,
        coeff_full,
        fused_weight_full,
        out_x_full,
        out_scale_full,
        NBLOCKS=nblocks,
        NORM_SCALE_=NORM_SCALE,
        BLOCK_K=FINAL_BLOCK_K,
        num_warps=4,
    )
    _finalize_kernel[(CHANNELS_SLICE,)](
        partial_x_slice,
        partial_x_rhs_slice,
        arg8_1,
        arg9_1,
        mean_term_slice,
        coeff_slice,
        fused_weight_slice,
        out_x_slice,
        out_scale_slice,
        NBLOCKS=nblocks,
        NORM_SCALE_=NORM_SCALE,
        BLOCK_K=FINAL_BLOCK_K,
        num_warps=4,
    )
    _pointwise_full_kernel[(triton.cdiv(TOTAL_FULL, POINTWISE_BLOCK),)](
        out_copy,
        arg2_1,
        arg3_1,
        mean_term_full,
        coeff_full,
        fused_weight_full,
        out_full,
        TOTAL_FULL_=TOTAL_FULL,
        CHANNELS_FULL_=CHANNELS_FULL,
        HW_=HW,
        WIDTH_=WIDTH,
        BLOCK=POINTWISE_BLOCK,
        num_warps=4,
    )
    _pointwise_slice_kernel[(triton.cdiv(TOTAL_SLICE, POINTWISE_BLOCK),)](
        out_copy,
        arg6_1,
        arg7_1,
        mean_term_slice,
        coeff_slice,
        fused_weight_slice,
        out_slice,
        TOTAL_SLICE_=TOTAL_SLICE,
        CHANNELS_FULL_=CHANNELS_FULL,
        CHANNELS_SLICE_=CHANNELS_SLICE,
        SLICE_START_=SLICE_START,
        HW_=HW,
        BLOCK=POINTWISE_BLOCK,
        num_warps=4,
    )

    return out_copy, out_x_full, out_scale_full, out_full, out_x_slice, out_scale_slice, out_slice
