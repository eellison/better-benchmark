"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 training-BatchNorm SiLU scope, including the bf16-to-fp32 statistics input, fp32 population var_mean over N,H,W, eps=1e-5 rsqrt side output, mutable running mean/variance copy_ aliases with the captured correction literal, affine scale/bias, Inductor's natural-exp SiLU cast placement, final bf16 channels-last activation, and all five returned outputs, whereas Inductor lowers the statistics, running-stat updates, SiLU, and stores as generic separated schedules; Inductor cannot do this today because its normalization scheduler does not fuse multi-output training-BatchNorm statistics, side-effecting running-stat updates, activation math, and direct channels-last output emission into one guarded plan; the fix is SCHEDULER_FUSION: add a training-BatchNorm SiLU scheduler template that preserves side outputs and mutable aliases while feeding the exact bf16 activation epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

from oracle_harness import oracle_impl


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
def _direct_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.arange(0, BLOCK_R)
    hw_size: tl.constexpr = H * W
    hw_idx = r_offsets - (r_offsets // hw_size) * hw_size
    n_idx = r_offsets // hw_size
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
    c_mask = channels < C
    r_mask = r_offsets < E
    mask = c_mask[:, None] & r_mask[None, :]
    x_offsets = (
        n_idx[None, :] * x_s0
        + channels[:, None] * x_s1
        + h_idx[None, :] * x_s2
        + w_idx[None, :] * x_s3
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(_f32_mul(x, x), axis=1)
    inv_e: tl.constexpr = 1.0 / E
    mean = _f32_mul(sum_x, inv_e)
    var = _f32_sub(_f32_mul(sum_x2, inv_e), _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    new_var = _f32_add(
        _f32_mul(_f32_mul(var, 1.0001220852154804), 0.1),
        _f32_mul(old_var, 0.9),
    )

    tl.store(mean_ptr + channels, mean, mask=c_mask)
    tl.store(invstd_ptr + channels, invstd, mask=c_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=c_mask)
    tl.store(running_var_ptr + channels, new_var, mask=c_mask)

@triton.jit
def _partial_sums_kernel(
    x_ptr,
    partial_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    hw_size: tl.constexpr = H * W
    hw_idx = r_offsets - (r_offsets // hw_size) * hw_size
    n_idx = r_offsets // hw_size
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
    c_mask = channels < C
    r_mask = r_offsets < E
    mask = c_mask[:, None] & r_mask[None, :]
    x_offsets = (
        n_idx[None, :] * x_s0
        + channels[:, None] * x_s1
        + h_idx[None, :] * x_s2
        + w_idx[None, :] * x_s3
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(_f32_mul(x, x), axis=1)
    offsets = tl.program_id(1) * C + channels
    plane = NUM_CHUNKS * C
    tl.store(partial_ptr + offsets, sum_x, mask=c_mask)
    tl.store(partial_ptr + plane + offsets, sum_x2, mask=c_mask)


@triton.jit
def _chunked_sum_stats_kernel(
    x_ptr,
    partial_ptr,
    C: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    r_offsets = tl.arange(0, BLOCK_R)
    mask_x = x_offsets < C * NUM_CHUNKS
    channels = x_offsets % C
    chunks = x_offsets // C
    x = tl.load(
        x_ptr + channels[:, None] + C * (chunks[:, None] * BLOCK_R + r_offsets[None, :]),
        mask=mask_x[:, None],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(x * x, axis=1)
    offsets = chunks * C + channels
    plane: tl.constexpr = NUM_CHUNKS * C
    tl.store(partial_ptr + offsets, sum_x, mask=mask_x)
    tl.store(partial_ptr + plane + offsets, sum_x2, mask=mask_x)

@triton.jit
def _finalize_stats_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    c_mask = channels < C
    mask = c_mask[:, None] & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]
    plane = NUM_CHUNKS * C
    sum_x = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=1)
    sum_x2 = tl.sum(tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=1)
    inv_e: tl.constexpr = 1.0 / E
    mean = _f32_mul(sum_x, inv_e)
    var = _f32_sub(_f32_mul(sum_x2, inv_e), _f32_mul(mean, mean))
    var = tl.where(var < 0.0, 0.0, var)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=c_mask, other=0.0).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    new_var = _f32_add(
        _f32_mul(_f32_mul(var, 1.0001220852154804), 0.1),
        _f32_mul(old_var, 0.9),
    )

    tl.store(mean_ptr + channels, mean, mask=c_mask)
    tl.store(invstd_ptr + channels, invstd, mask=c_mask)
    tl.store(running_mean_ptr + channels, new_mean, mask=c_mask)
    tl.store(running_var_ptr + channels, new_var, mask=c_mask)


@triton.jit
def _silu_epilogue_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    if CHANNELS_LAST:
        channels = offsets - (offsets // C) * C
    else:
        channels = (offsets // HW) - ((offsets // HW) // C) * C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    norm = (x - mean) * invstd
    affine = norm * weight + bias
    denom = tl_math.exp(-affine) + 1.0
    silu = affine / denom
    tl.store(out_ptr + offsets, silu.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="000da78a", DIRECT=0, DIRECT_BLOCK_C=8, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=1, ACT_BLOCK=1024, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4, CHUNK_SUM=1, CHUNK_X=16)
@oracle_impl(hardware="B200", point="2e27daef", DIRECT=1, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="ecabde94", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="3a7a99d6", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="85dd3e54", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="ceebf62c", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=2048, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="ddcb5e92", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=2048, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="d22d384a", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=2048, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="42a503b0", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="2814ad41", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="e9256d98", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="0f9a93bc", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="837ab064", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="c2791544", DIRECT=1, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="e41460a6", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="886f27c9", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=1024, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="cdca2f80", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=2048, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="46238279", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
@oracle_impl(hardware="B200", point="bf1cc8fb", DIRECT=0, DIRECT_BLOCK_C=4, STATS_BLOCK_R=4096, STATS_BLOCK_C=8, FINAL_BLOCK_C=8, ACT_BLOCK=512, direct_warps=8, stats_warps=4, final_warps=1, act_warps=4)
def oracle_forward(
    inputs,
    *,
    DIRECT: int,
    DIRECT_BLOCK_C: int,
    STATS_BLOCK_R: int,
    STATS_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    ACT_BLOCK: int,
    direct_warps: int,
    stats_warps: int,
    final_warps: int,
    act_warps: int,
    CHUNK_SUM: int = 0,
    CHUNK_X: int = 16,
):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
    total = n * c * hw

    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    if CHUNK_SUM:
        num_chunks = triton.cdiv(e, 128)
        partial = torch.empty((2, num_chunks, c), device=x.device, dtype=torch.float32)
        _chunked_sum_stats_kernel[(triton.cdiv(c * num_chunks, CHUNK_X),)](
            x,
            partial,
            C=c,
            NUM_CHUNKS=num_chunks,
            BLOCK_R=128,
            BLOCK_X=CHUNK_X,
            num_warps=stats_warps,
            num_stages=3,
        )
        _finalize_stats_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            partial,
            running_mean,
            running_var,
            mean,
            invstd,
            C=c,
            E=e,
            NUM_CHUNKS=num_chunks,
            BLOCK_CHUNKS=_next_power_of_2(num_chunks),
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=final_warps,
            num_stages=3,
        )
    elif DIRECT:
        direct_block_r = triton.next_power_of_2(e)
        _direct_stats_kernel[(triton.cdiv(c, DIRECT_BLOCK_C),)](
            x,
            running_mean,
            running_var,
            mean,
            invstd,
            C=c,
            H=h,
            W=w,
            E=e,
            x_s0=x.stride(0),
            x_s1=x.stride(1),
            x_s2=x.stride(2),
            x_s3=x.stride(3),
            BLOCK_C=DIRECT_BLOCK_C,
            BLOCK_R=direct_block_r,
            num_warps=direct_warps,
            num_stages=3,
        )
    else:
        num_chunks = triton.cdiv(e, STATS_BLOCK_R)
        partial = torch.empty((2, num_chunks, c), device=x.device, dtype=torch.float32)
        _partial_sums_kernel[(triton.cdiv(c, STATS_BLOCK_C), num_chunks)](
            x,
            partial,
            C=c,
            H=h,
            W=w,
            E=e,
            x_s0=x.stride(0),
            x_s1=x.stride(1),
            x_s2=x.stride(2),
            x_s3=x.stride(3),
            NUM_CHUNKS=num_chunks,
            BLOCK_C=STATS_BLOCK_C,
            BLOCK_R=STATS_BLOCK_R,
            num_warps=stats_warps,
            num_stages=3,
        )
        _finalize_stats_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            partial,
            running_mean,
            running_var,
            mean,
            invstd,
            C=c,
            E=e,
            NUM_CHUNKS=num_chunks,
            BLOCK_CHUNKS=_next_power_of_2(num_chunks),
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=final_warps,
            num_stages=3,
        )

    _silu_epilogue_kernel[(triton.cdiv(total, ACT_BLOCK),)](
        x,
        weight,
        bias,
        mean,
        invstd,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        CHANNELS_LAST=x.stride(1) == 1,
        BLOCK=ACT_BLOCK,
        num_warps=act_warps,
        num_stages=3,
    )
    return mean, invstd, out, running_mean, running_var
