"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 add/copy/slice BatchNorm-backward fragment by materializing both returned add layouts, cooperatively accumulating the sibling `sum([0,2,3])` channel reductions over the sliced second half, finalizing the f32 channel summaries once, and sinking them into the returned channels-last bf16 dense epilogue, whereas Inductor schedules the memory-format copy/clone/copy path, slice reductions, dependent vector multiply, and full tensor epilogue as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates layout-changing side outputs, overlapping sliced channel reductions, and dependent dense stores while preserving the bf16 add rounding boundary; the fix is COOPERATIVE_SPLIT_K: add a guarded split-K BN-backward lowering that shares partial channel reductions from the layout producer and emits the required vector and layout-distinct tensor outputs together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 3.985969387755102e-05
FULL_C = 160
TAIL_C = 80
HW = 49
EPILOGUE_BLOCK = 512


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _copy_add_kernel(
    x0_ptr,
    x1_ptr,
    clone_ptr,
    copy_ptr,
    K_TOTAL: tl.constexpr,
    C_START: tl.constexpr,
    C_COUNT: tl.constexpr,
    FULL_C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k_offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_local = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_offsets = C_START + c_local
    k_mask = k_offsets < K_TOTAL
    c_mask = c_local < C_COUNT

    n = k_offsets // HW_
    hw = k_offsets - n * HW_
    contiguous_offsets = n[:, None] * FULL_C_ * HW_ + c_offsets[None, :] * HW_ + hw[:, None]
    channels_last_offsets = n[:, None] * FULL_C_ * HW_ + hw[:, None] * FULL_C_ + c_offsets[None, :]
    mask = k_mask[:, None] & c_mask[None, :]

    x0 = tl.load(x0_ptr + contiguous_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + channels_last_offsets, mask=mask, other=0.0).to(tl.float32)
    added_bf16 = (x0 + x1).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(clone_ptr + contiguous_offsets, added_bf16, mask=mask)
    tl.store(copy_ptr + channels_last_offsets, added_bf16, mask=mask)


@triton.jit
def _tail_copy_and_partial_reduce_kernel(
    x0_ptr,
    x1_ptr,
    rhs_ptr,
    mean_ptr,
    clone_ptr,
    copy_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    K_TOTAL: tl.constexpr,
    FULL_C_: tl.constexpr,
    TAIL_C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k_offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    tail = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_offsets = tail + TAIL_C_
    k_mask = k_offsets < K_TOTAL
    c_mask = tail < TAIL_C_

    n = k_offsets // HW_
    hw = k_offsets - n * HW_
    contiguous_offsets = n[:, None] * FULL_C_ * HW_ + c_offsets[None, :] * HW_ + hw[:, None]
    channels_last_offsets = n[:, None] * FULL_C_ * HW_ + hw[:, None] * FULL_C_ + c_offsets[None, :]
    mask = k_mask[:, None] & c_mask[None, :]

    x0 = tl.load(x0_ptr + contiguous_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + channels_last_offsets, mask=mask, other=0.0).to(tl.float32)
    added_bf16 = (x0 + x1).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(clone_ptr + contiguous_offsets, added_bf16, mask=mask)
    tl.store(copy_ptr + channels_last_offsets, added_bf16, mask=mask)

    tail_offsets = n[:, None] * TAIL_C_ * HW_ + hw[:, None] * TAIL_C_ + tail[None, :]
    reduce_mask = k_mask[:, None] & c_mask[None, :]

    x_tail = added_bf16.to(tl.float32)
    rhs = tl.load(rhs_ptr + tail_offsets, mask=reduce_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + tail, mask=c_mask, other=0.0).to(tl.float32)
    centered = rhs - mean[None, :]
    dot = x_tail * centered

    partial_sum = tl.sum(tl.where(reduce_mask, x_tail, 0.0), axis=0)
    partial_dot = tl.sum(tl.where(reduce_mask, dot, 0.0), axis=0)
    partial_offsets = tl.program_id(0) * TAIL_C_ + tail
    tl.store(partial_sum_ptr + partial_offsets, partial_sum, mask=c_mask)
    tl.store(partial_dot_ptr + partial_offsets, partial_dot, mask=c_mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    weight_ptr,
    gain_ptr,
    sum_out_ptr,
    mul8_out_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    gain_coeff_ptr,
    NUM_K_TILES: tl.constexpr,
    TAIL_C_: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    c_mask = c_offsets < TAIL_C_
    tile_mask = tile_offsets < NUM_K_TILES
    offsets = tile_offsets[:, None] * TAIL_C_ + c_offsets[None, :]
    mask = tile_mask[:, None] & c_mask[None, :]

    sum1 = tl.sum(tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    sum2 = tl.sum(tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    gain = tl.load(gain_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    mul2 = sum2 * scale
    mul3 = weight * weight
    mul4 = mul2 * mul3
    mul5 = weight * gain
    mul8 = sum2 * weight

    tl.store(sum_out_ptr + c_offsets, sum1, mask=c_mask)
    tl.store(mul8_out_ptr + c_offsets, mul8, mask=c_mask)
    tl.store(mean_term_ptr + c_offsets, sum1 * scale, mask=c_mask)
    tl.store(dot_coeff_ptr + c_offsets, mul4, mask=c_mask)
    tl.store(gain_coeff_ptr + c_offsets, mul5, mask=c_mask)


@triton.jit
def _epilogue_kernel(
    copy_ptr,
    rhs_ptr,
    mean_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    gain_coeff_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    FULL_C_: tl.constexpr,
    TAIL_C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c = offsets % TAIL_C_
    row = offsets // TAIL_C_
    n = row // HW_
    hw = row - n * HW_
    copy_offsets = n * FULL_C_ * HW_ + hw * FULL_C_ + (c + TAIL_C_)

    x = tl.load(copy_ptr + copy_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    centered = rhs - mean
    dot_coeff = tl.load(dot_coeff_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(mean_term_ptr + c, mask=mask, other=0.0).to(tl.float32)
    gain_coeff = tl.load(gain_coeff_ptr + c, mask=mask, other=0.0).to(tl.float32)

    result = (x - centered * dot_coeff - mean_term) * gain_coeff
    tl.store(out_ptr + offsets, result.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(hardware="B200", point="e2841909", BLOCK_K=256, BLOCK_C=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C, num_warps):
    x0, x1, rhs, mean, weight, gain, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    batch = int(x0.shape[0])
    full_c = int(x0.shape[1])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    tail_c = full_c // 2
    k_total = batch * height * width
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    clone = torch.empty_strided(
        (batch, full_c, height, width),
        (full_c * height * width, height * width, width, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    copy = torch.empty_strided(
        (batch, full_c, height, width),
        (full_c * height * width, 1, width * full_c, full_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((tail_c,), device=x0.device, dtype=torch.float32)
    mul8_out = torch.empty((tail_c,), device=x0.device, dtype=torch.float32)
    mean_term = torch.empty((tail_c,), device=x0.device, dtype=torch.float32)
    dot_coeff = torch.empty((tail_c,), device=x0.device, dtype=torch.float32)
    gain_coeff = torch.empty((tail_c,), device=x0.device, dtype=torch.float32)
    partial_sum = torch.empty((num_k_tiles, tail_c), device=x0.device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, tail_c), device=x0.device, dtype=torch.float32)
    out = torch.empty_strided(
        (batch, tail_c, height, width),
        (tail_c * height * width, 1, width * tail_c, tail_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    _copy_add_kernel[(num_k_tiles, triton.cdiv(tail_c, BLOCK_C))](
        x0,
        x1,
        clone,
        copy,
        K_TOTAL=k_total,
        C_START=0,
        C_COUNT=tail_c,
        FULL_C_=full_c,
        HW_=height * width,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    _tail_copy_and_partial_reduce_kernel[(num_k_tiles, triton.cdiv(tail_c, BLOCK_C))](
        x0,
        x1,
        rhs,
        mean,
        clone,
        copy,
        partial_sum,
        partial_dot,
        K_TOTAL=k_total,
        FULL_C_=full_c,
        TAIL_C_=tail_c,
        HW_=height * width,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(tail_c, BLOCK_C),)](
        partial_sum,
        partial_dot,
        weight,
        gain,
        sum_out,
        mul8_out,
        mean_term,
        dot_coeff,
        gain_coeff,
        NUM_K_TILES=num_k_tiles,
        TAIL_C_=tail_c,
        scale=SCALE,
        BLOCK_TILES=block_tiles,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(out.numel(), EPILOGUE_BLOCK),)](
        copy,
        rhs,
        mean,
        mean_term,
        dot_coeff,
        gain_coeff,
        out,
        TOTAL=out.numel(),
        FULL_C_=full_c,
        TAIL_C_=tail_c,
        HW_=height * width,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return clone, copy, sum_out, mul8_out, out
