"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Adv-Inception training-BatchNorm maxpool fanout, including the fp32 population `var_mean(..., correction=0)` over channels-last input, eps=0.001 rsqrt side output, mutable running-stat copy_ aliases with the captured correction literal, exact fp32 affine followed by bf16 rounding and NaN-preserving ReLU, returned 3x3 stride-2 low-memory maxpool values and int8 offsets, and the returned padded 3x3 avg_pool2d over the maxpool tensor, whereas Inductor lowers the BN statistics, running-stat epilogue, rounded ReLU activation, maxpool tuple, and avgpool consumer as separate generic schedules; Inductor cannot do this today because its scheduler does not keep training-BN reductions with mutable aliases virtual through a downstream low-memory pooling stencil while preserving visible side outputs and bf16 cast boundaries; the fix is SCHEDULER_FUSION: extend the training-BatchNorm and pooling schedulers to co-schedule stats, aliases, affine/ReLU, maxpool offsets, and fixed avgpool consumers in one full-scope plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 192
H_IN = 71
W_IN = 71
H_OUT = 35
W_OUT = 35
IN_HW = H_IN * W_IN
OUT_HW = H_OUT * W_OUT
ELEM_PER_CHANNEL = BATCH * IN_HW
RUNNING_VAR_CORRECTION = 1.00000154979411


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
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return tl.where(x != x, x, rounded.to(tl.float32, bitcast=True))


@triton.jit
def _bn_partial_stats_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sum2_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    e_block = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    e_offsets = e_block * BLOCK_E + tl.arange(0, BLOCK_E)

    hw = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = (
        n_idx[None, :] * SN
        + channels[:, None] * SC
        + h_idx[None, :] * SH
        + w_idx[None, :] * SW
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    vals = tl.where(mask, vals, 0.0)

    sums = tl.sum(vals, axis=1)
    sums2 = tl.sum(_f32_mul(vals, vals), axis=1)
    out_offsets = e_block * C + channels
    tl.store(partial_sum_ptr + out_offsets, sums, mask=channels < C)
    tl.store(partial_sum2_ptr + out_offsets, sums2, mask=channels < C)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_sum_ptr,
    partial_sum2_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    CORRECTION: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]

    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
    sums2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(sums, axis=1).to(tl.float32)
    total2 = tl.sum(sums2, axis=1).to(tl.float32)
    mean = _f32_mul(total, 1.0 / E)
    var = _f32_sub(_f32_mul(total2, 1.0 / E), _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 0.001))
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(old_mean, 0.9), _f32_mul(mean, 0.1))
    new_var = _f32_add(
        _f32_mul(old_var, 0.9),
        _f32_mul(_f32_mul(var, CORRECTION), 0.1),
    )

    tl.store(mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, new_var, mask=channel_mask)


@triton.jit
def _bn_relu_maxpool_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    pool_ptr,
    offsets_ptr,
    C: tl.constexpr,
    OUT_HW_: tl.constexpr,
    W_OUT_: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    s_offsets = tl.program_id(2) * BLOCK_S + tl.arange(0, BLOCK_S)
    c_mask = c_offsets < C
    s_mask = s_offsets < OUT_HW_

    out_h = s_offsets // W_OUT_
    out_w = s_offsets - out_h * W_OUT_
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    best = tl.full((BLOCK_S, BLOCK_C), -float("inf"), tl.float32)
    best_idx = tl.zeros((BLOCK_S, BLOCK_C), tl.int32)

    for kh in tl.static_range(0, 3):
        in_h = out_h * 2 + kh
        for kw in tl.static_range(0, 3):
            in_w = out_w * 2 + kw
            x_offsets = (
                batch * SN
                + c_offsets[None, :] * SC
                + in_h[:, None] * SH
                + in_w[:, None] * SW
            )
            mask = s_mask[:, None] & c_mask[None, :]
            x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
            affine = _f32_add(
                _f32_mul(_f32_mul(_f32_sub(x, mean[None, :]), invstd[None, :]), weight[None, :]),
                bias[None, :],
            )
            rounded = _round_bf16_to_fp32(affine)
            relu = tl.where((rounded > 0.0) | (rounded != rounded), rounded, 0.0)

            take = mask & ((relu > best) | (relu != relu))
            best = tl.where(take, relu, best)
            best_idx = tl.where(take, kh * 3 + kw, best_idx)

    out_offsets = batch * (C * OUT_HW_) + s_offsets[:, None] * C + c_offsets[None, :]
    out_mask = s_mask[:, None] & c_mask[None, :]
    tl.store(pool_ptr + out_offsets, best, mask=out_mask)
    tl.store(offsets_ptr + out_offsets, best_idx.to(tl.int8), mask=out_mask)


@triton.jit
def _avgpool3x3_kernel(
    pool_ptr,
    avg_ptr,
    C: tl.constexpr,
    H_OUT_: tl.constexpr,
    W_OUT_: tl.constexpr,
    OUT_HW_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    s_offsets = tl.program_id(2) * BLOCK_S + tl.arange(0, BLOCK_S)
    c_mask = c_offsets < C
    s_mask = s_offsets < OUT_HW_

    out_h = s_offsets // W_OUT_
    out_w = s_offsets - out_h * W_OUT_
    batch_base = batch * (C * OUT_HW_)
    acc = tl.zeros((BLOCK_S, BLOCK_C), tl.float32)

    for kh in tl.static_range(0, 3):
        in_h = out_h + kh - 1
        valid_h = (in_h >= 0) & (in_h < H_OUT_)
        for kw in tl.static_range(0, 3):
            in_w = out_w + kw - 1
            valid = s_mask & valid_h & (in_w >= 0) & (in_w < W_OUT_)
            offsets = batch_base + (in_h[:, None] * W_OUT_ + in_w[:, None]) * C + c_offsets[None, :]
            vals = tl.load(pool_ptr + offsets, mask=valid[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
            acc = _f32_add(acc, vals)

    out_offsets = batch_base + s_offsets[:, None] * C + c_offsets[None, :]
    avg = _round_bf16_to_fp32(_f32_mul(acc, 1.0 / 9.0))
    tl.store(avg_ptr + out_offsets, avg, mask=s_mask[:, None] & c_mask[None, :])


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="9b65830c",
    BLOCK_E=4096,
    STATS_BLOCK_C=8,
    POOL_BLOCK_C=64,
    POOL_BLOCK_S=16,
    num_warps_stats=8,
    num_warps_pool=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    STATS_BLOCK_C: int,
    POOL_BLOCK_C: int,
    POOL_BLOCK_S: int,
    num_warps_stats: int,
    num_warps_pool: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    num_chunks = triton.cdiv(ELEM_PER_CHANNEL, BLOCK_E)
    block_chunks = _next_power_of_2(num_chunks)

    partial_sum = torch.empty((num_chunks, CHANNELS), device=arg0_1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, CHANNELS), device=arg0_1.device, dtype=torch.float32)
    mean = torch.empty_strided((1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    out_shape = (BATCH, CHANNELS, H_OUT, W_OUT)
    out_stride = (CHANNELS * OUT_HW, 1, W_OUT * CHANNELS, CHANNELS)
    pool = torch.empty_strided(out_shape, out_stride, device=arg0_1.device, dtype=torch.bfloat16)
    pool_offsets = torch.empty_strided(out_shape, out_stride, device=arg0_1.device, dtype=torch.int8)
    avg = torch.empty_strided(out_shape, out_stride, device=arg0_1.device, dtype=torch.bfloat16)

    _bn_partial_stats_kernel[(triton.cdiv(CHANNELS, STATS_BLOCK_C), num_chunks)](
        arg0_1,
        partial_sum,
        partial_sum2,
        C=CHANNELS,
        H=H_IN,
        W=W_IN,
        E=ELEM_PER_CHANNEL,
        SN=int(arg0_1.stride(0)),
        SC=int(arg0_1.stride(1)),
        SH=int(arg0_1.stride(2)),
        SW=int(arg0_1.stride(3)),
        BLOCK_E=BLOCK_E,
        BLOCK_C=STATS_BLOCK_C,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _bn_finalize_stats_kernel[(triton.cdiv(CHANNELS, STATS_BLOCK_C),)](
        partial_sum,
        partial_sum2,
        arg1_1,
        arg2_1,
        mean,
        invstd,
        C=CHANNELS,
        E=ELEM_PER_CHANNEL,
        CORRECTION=RUNNING_VAR_CORRECTION,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=STATS_BLOCK_C,
        num_warps=4,
        num_stages=3,
    )

    pool_grid = (BATCH, triton.cdiv(CHANNELS, POOL_BLOCK_C), triton.cdiv(OUT_HW, POOL_BLOCK_S))
    _bn_relu_maxpool_kernel[pool_grid](
        arg0_1,
        mean,
        invstd,
        arg3_1,
        arg4_1,
        pool,
        pool_offsets,
        C=CHANNELS,
        OUT_HW_=OUT_HW,
        W_OUT_=W_OUT,
        SN=int(arg0_1.stride(0)),
        SC=int(arg0_1.stride(1)),
        SH=int(arg0_1.stride(2)),
        SW=int(arg0_1.stride(3)),
        BLOCK_C=POOL_BLOCK_C,
        BLOCK_S=POOL_BLOCK_S,
        num_warps=num_warps_pool,
        num_stages=3,
    )
    _avgpool3x3_kernel[pool_grid](
        pool,
        avg,
        C=CHANNELS,
        H_OUT_=H_OUT,
        W_OUT_=W_OUT,
        OUT_HW_=OUT_HW,
        BLOCK_C=POOL_BLOCK_C,
        BLOCK_S=POOL_BLOCK_S,
        num_warps=num_warps_pool,
        num_stages=3,
    )
    return mean, invstd, pool, pool_offsets, avg, arg1_1, arg2_1
