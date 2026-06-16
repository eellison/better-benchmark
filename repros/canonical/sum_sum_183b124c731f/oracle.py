"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle cooperatively splits the shared `N,H,W` channel domain, co-reduces `sum(arg0)` and `sum(arg0 * (arg1 - mean))`, finalizes the returned fp32 channel vectors, and uses those finalized summaries to emit the full channels-last bf16 BatchNorm-backward tensor, whereas Inductor schedules the sibling channel reductions and dependent dense epilogue as generic reduction/pointwise work around separate graph regions; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned cooperative split-K template that coordinates multiple channel reductions with vector and full-tensor epilogues while preserving bf16-to-fp32 input casts and final bf16 rounding; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward channel-reduction lowering that splits the reduced `N,H,W` domain, combines sibling summaries once, and sinks them into the dependent vector and dense epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 6.228077168367346e-07
EPILOGUE_BLOCK = 1024


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets = k_offsets[:, None] * C + c_offsets[None, :]
    mask = (k_offsets[:, None] < K_TOTAL) & (c_offsets[None, :] < C)

    grad = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = activation - mean[None, :]

    partial_sum = tl.sum(tl.where(mask, grad, 0.0), axis=0)
    partial_dot = tl.sum(tl.where(mask, grad * centered, 0.0), axis=0)
    out_offsets = tl.program_id(1) * C + c_offsets
    tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=c_offsets < C)
    tl.store(partial_dot_ptr + out_offsets, partial_dot, mask=c_offsets < C)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    out_sum_ptr,
    sum_dot_ptr,
    out_vec_ptr,
    C: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_K_TILES
    offsets = tile_offsets * C + c

    sum1 = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum2 = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(out_sum_ptr + c, sum1)
    tl.store(sum_dot_ptr + c, sum2)
    tl.store(out_vec_ptr + c, sum2 * invstd)


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c = offsets % C

    grad = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = activation - mean
    mean_term = sum1 * SCALE_VALUE
    scaled_dot = sum2 * SCALE_VALUE
    invstd_sq = invstd * invstd
    correction = scaled_dot * invstd_sq
    centered_correction = centered * correction
    sub_1 = grad - centered_correction
    sub_2 = sub_1 - mean_term
    out_scale = invstd * weight
    result = sub_2 * out_scale
    tl.store(out_ptr + offsets, result, mask=mask)


@oracle_impl(hardware="B200", point="806629ba", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="cbe8cdcd", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="14bc6292", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="bc4d6b09", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="27df89f8", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="b330737b", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="99caacbc", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="06e3c268", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="7f09d5e1", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="1c020ea5", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="51768188", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="0b7b8941", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="4a86e94b", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="797c887f", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="fa717ebf", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="64702d33", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="fbff83c8", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="c63234aa", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="0ce2ed98", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="a999a033", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="59332feb", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="0b3636fa", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="05aad2cb", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="92d95c9a", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="feb664f7", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="ba1f5763", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="3ab290cb", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="c37e4be5", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="7ef2067a", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="b38c4fd4", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="b2f8b132", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="bcf0f34b", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="45be9d7c", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="8f8e9775", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="16d83ade", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="3c4c0e95", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="56236213", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="73cdbfa1", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="e001e4c0", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="4248d2e3", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="805fe5b3", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="86433bd5", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="819a1512", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="8a1d8530", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="40a8e8e1", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, num_warps: int):
    arg0, arg1, arg2, arg3, arg4 = inputs
    n, c, h, w = arg0.shape
    k_total = n * h * w
    total = arg0.numel()
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    partial_sum = torch.empty((num_k_tiles, c), device=arg0.device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, c), device=arg0.device, dtype=torch.float32)
    out_sum = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    sum_dot = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out_vec = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=arg0.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        arg0,
        arg1,
        arg2,
        partial_sum,
        partial_dot,
        C=c,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        arg3,
        out_sum,
        sum_dot,
        out_vec,
        C=c,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        out_sum,
        sum_dot,
        out,
        TOTAL=total,
        C=c,
        SCALE_VALUE=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return out_sum, out_vec, out
