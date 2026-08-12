"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Inception two-branch training-BatchNorm fanout plus independent low-memory maxpool branch, including both fp32 population `var_mean(..., correction=0)` reductions over channels-last inputs, eps=0.001 rsqrt side outputs, four mutable running-stat `copy_` updates with the captured correction literal, bf16 affine/ReLU cast boundaries, int8 low-memory maxpool offsets, the returned channels-last bf16 channel cat, and the padded 3x3 avg_pool2d tail, whereas Inductor lowers the sibling BN-training branches, mutable copy_ epilogues, maxpool-with-offsets branch, channel cat, and pooling consumer as separate generic schedules; Inductor cannot do this today because the scheduler does not co-schedule independent training-BN templates with mutable aliases while sinking a multi-output maxpool branch, static channel-cat layout, and avg-pool tail into one full-scope plan; the fix is SCHEDULER_FUSION: extend the training-BatchNorm scheduler to group sibling channel-stat reductions, emit running-stat side effects, and fuse exact maxpool/cat/pool materialization for static Inception branch fanouts."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 0.001
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804


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
def _pair_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    partial_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    TOTAL_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    S0N: tl.constexpr,
    S0C: tl.constexpr,
    S0H: tl.constexpr,
    S0W: tl.constexpr,
    S1N: tl.constexpr,
    S1C: tl.constexpr,
    S1H: tl.constexpr,
    S1W: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    RAW_STATS: tl.constexpr,
):
    c_block = tl.program_id(0)
    chunk = tl.program_id(1)
    branch = tl.program_id(2)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    r_offsets = chunk * BLOCK_R + tl.arange(0, BLOCK_R)

    hw = r_offsets % (H * W)
    n_idx = r_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    r_live = r_offsets < E

    is0 = branch == 0
    is1 = branch == 1
    c_live0 = channels < C0
    c_live1 = channels < C1
    live0 = is0 & c_live0
    live1 = is1 & c_live1
    channel_live = live0 | live1
    active = channel_live[:, None] & r_live[None, :]

    off0 = (
        n_idx[None, :] * S0N
        + channels[:, None] * S0C
        + h_idx[None, :] * S0H
        + w_idx[None, :] * S0W
    )
    off1 = (
        n_idx[None, :] * S1N
        + channels[:, None] * S1C
        + h_idx[None, :] * S1H
        + w_idx[None, :] * S1W
    )

    vals0 = tl.load(x0_ptr + off0, mask=live0[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals1 = tl.load(x1_ptr + off1, mask=live1[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals = tl.where(active, _f32_add(vals0, vals1), 0.0)
    if RAW_STATS:
        partial0 = tl.sum(vals, axis=1)
        partial1 = tl.sum(_f32_mul(vals, vals), axis=1)
    else:
        weights = tl.where(active, 1.0, 0.0)
        zero = tl.zeros((BLOCK_C, BLOCK_R), tl.float32)
        partial0, partial1, _ = triton_helpers.welford(vals, zero, weights, 1)

    global_channels = tl.where(is0, channels, C0 + channels)
    plane = NUM_CHUNKS * TOTAL_C
    store_offsets = chunk * TOTAL_C + global_channels
    tl.store(partial_ptr + store_offsets, partial0, mask=channel_live)
    tl.store(partial_ptr + plane + store_offsets, partial1, mask=channel_live)


@triton.jit
def _pair_finalize_stats_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    mean0_ptr,
    invstd0_ptr,
    mean1_ptr,
    invstd1_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    TOTAL_C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    RAW_STATS: tl.constexpr,
):
    c_block = tl.program_id(0)
    branch = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, BLOCK_CHUNKS)

    is0 = branch == 0
    is1 = branch == 1
    c_live0 = channels < C0
    c_live1 = channels < C1
    channel_live = (is0 & c_live0) | (is1 & c_live1)
    global_channels = tl.where(is0, channels, C0 + channels)

    mask = channel_live[:, None] & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * TOTAL_C + global_channels[:, None]
    plane = NUM_CHUNKS * TOTAL_C
    chunk0 = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    chunk1 = tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0)
    if RAW_STATS:
        total = tl.sum(chunk0, axis=1).to(tl.float32)
        total2 = tl.sum(chunk1, axis=1).to(tl.float32)
        mean = _f32_mul(total, 1.0 / E)
        var = _f32_sub(_f32_mul(total2, 1.0 / E), _f32_mul(mean, mean))
        var = tl.where(var < 0.0, 0.0, var)
    else:
        chunk_start = chunks * BLOCK_R
        counts = tl.where(chunks < NUM_CHUNKS, tl.minimum(BLOCK_R, E - chunk_start), 0).to(tl.float32)
        weights = tl.broadcast_to(counts[None, :], (BLOCK_C, BLOCK_CHUNKS))
        mean, m2, _ = triton_helpers.welford(chunk0, chunk1, weights, 1)
        var = _f32_mul(m2, 1.0 / E)
    invstd = libdevice.rsqrt(_f32_add(var, 0.001))

    old_mean0 = tl.load(running_mean0_ptr + channels, mask=is0 & c_live0, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + channels, mask=is0 & c_live0, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + channels, mask=is1 & c_live1, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + channels, mask=is1 & c_live1, other=0.0).to(tl.float32)

    new_mean0 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean0, 0.9))
    new_var0 = _f32_add(
        _f32_mul(_f32_mul(var, 1.0001220852154804), 0.1),
        _f32_mul(old_var0, 0.9),
    )
    new_mean1 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean1, 0.9))
    new_var1 = _f32_add(
        _f32_mul(_f32_mul(var, 1.0001220852154804), 0.1),
        _f32_mul(old_var1, 0.9),
    )

    tl.store(mean0_ptr + channels, mean, mask=is0 & c_live0)
    tl.store(invstd0_ptr + channels, invstd, mask=is0 & c_live0)
    tl.store(running_mean0_ptr + channels, new_mean0, mask=is0 & c_live0)
    tl.store(running_var0_ptr + channels, new_var0, mask=is0 & c_live0)

    tl.store(mean1_ptr + channels, mean, mask=is1 & c_live1)
    tl.store(invstd1_ptr + channels, invstd, mask=is1 & c_live1)
    tl.store(running_mean1_ptr + channels, new_mean1, mask=is1 & c_live1)
    tl.store(running_var1_ptr + channels, new_var1, mask=is1 & c_live1)


@triton.jit
def _bn_relu_cat_branch_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    OUT_OFFSET: tl.constexpr,
    C: tl.constexpr,
    TOTAL_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channel = offsets % C
    tmp = offsets // C
    w_idx = tmp % W
    tmp = tmp // W
    h_idx = tmp % H
    n_idx = tmp // H

    input_offsets = n_idx * SN + channel * SC + h_idx * SH + w_idx * SW
    x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    affine = _f32_add(_f32_mul(_f32_mul(_f32_sub(x, mean), invstd), weight), bias)
    rounded = affine.to(tl.bfloat16)
    relu = tl.where((rounded > 0.0) | (rounded != rounded), rounded, 0.0)

    store_offsets = ((n_idx * H + h_idx) * W + w_idx) * TOTAL_C + OUT_OFFSET + channel
    tl.store(out_ptr + store_offsets, relu, mask=mask)


@triton.jit
def _maxpool_cat_kernel(
    x_ptr,
    cat_ptr,
    offsets_ptr,
    OUT_OFFSET: tl.constexpr,
    C: tl.constexpr,
    TOTAL_C: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    TOTAL: tl.constexpr,
    SN: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    channel = offsets % C
    tmp = offsets // C
    ow = tmp % W_OUT
    tmp = tmp // W_OUT
    oh = tmp % H_OUT
    n_idx = tmp // H_OUT

    best = tl.full((BLOCK,), -float("inf"), tl.float32)
    best_offset = tl.zeros((BLOCK,), tl.int32)
    for kh in tl.static_range(0, 3):
        ih = oh * 2 + kh
        for kw in tl.static_range(0, 3):
            iw = ow * 2 + kw
            valid = mask & (ih < H_IN) & (iw < W_IN)
            load_offsets = n_idx * SN + channel * SC + ih * SH + iw * SW
            val = tl.load(x_ptr + load_offsets, mask=valid, other=0.0).to(tl.float32)
            take = valid & ((val > best) | (val != val))
            best = tl.where(take, val, best)
            best_offset = tl.where(take, kh * 3 + kw, best_offset)

    cat_offsets = ((n_idx * H_OUT + oh) * W_OUT + ow) * TOTAL_C + OUT_OFFSET + channel
    pool_offsets = ((n_idx * H_OUT + oh) * W_OUT + ow) * C + channel
    tl.store(cat_ptr + cat_offsets, best.to(tl.bfloat16), mask=mask)
    tl.store(offsets_ptr + pool_offsets, best_offset.to(tl.int8), mask=mask)


@triton.jit
def _avg_pool3x3_kernel(
    cat_ptr,
    pool_ptr,
    TOTAL_C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    out_c = offsets % TOTAL_C
    tmp = offsets // TOTAL_C
    ow = tmp % W
    tmp = tmp // W
    oh = tmp % H
    n_idx = tmp // H

    acc = tl.zeros((BLOCK,), tl.float32)
    for kh in tl.static_range(0, 3):
        ih = oh + kh - 1
        valid_h = (ih >= 0) & (ih < H)
        for kw in tl.static_range(0, 3):
            iw = ow + kw - 1
            valid = mask & valid_h & (iw >= 0) & (iw < W)
            load_offsets = ((n_idx * H + ih) * W + iw) * TOTAL_C + out_c
            vals = tl.load(cat_ptr + load_offsets, mask=valid, other=0.0).to(tl.float32)
            acc = _f32_add(acc, tl.where(valid, vals, 0.0))

    pooled = _f32_mul(acc, 1.0 / 9.0).to(tl.bfloat16)
    tl.store(pool_ptr + offsets, pooled, mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _run(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    CAT_BLOCK: int,
    POOL_BRANCH_BLOCK: int,
    AVG_BLOCK: int,
    num_warps_stats: int,
    num_warps_final: int,
    RAW_STATS: bool,
):
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        pool_x,
        _shape_param_0,
        _shape_param_1,
    ) = inputs

    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    pool_c = int(pool_x.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    pool_h_in = int(pool_x.shape[2])
    pool_w_in = int(pool_x.shape[3])
    e = n * h * w
    stats_c = c0 + c1
    total_c = stats_c + pool_c
    max_stats_c = max(c0, c1)
    total = n * total_c * h * w
    pool_branch_total = n * pool_c * h * w
    num_chunks = triton.cdiv(e, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    partial = torch.empty((2, num_chunks, stats_c), device=x0.device, dtype=torch.float32)
    mean0 = torch.empty_strided((1, c0, 1, 1), (c0, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd0 = torch.empty_strided((1, c0, 1, 1), (c0, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, c1, 1, 1), (c1, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty_strided((1, c1, 1, 1), (c1, 1, 1, 1), device=x0.device, dtype=torch.float32)
    cat = torch.empty_strided(
        (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    pool_offsets = torch.empty_strided(
        (n, pool_c, h, w),
        (pool_c * h * w, 1, pool_c * w, pool_c),
        device=x0.device,
        dtype=torch.int8,
    )
    avg_pool = torch.empty_strided(
        (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    _pair_partial_stats_kernel[(triton.cdiv(max_stats_c, BLOCK_C), num_chunks, 2)](
        x0,
        x1,
        partial,
        C0=c0,
        C1=c1,
        TOTAL_C=stats_c,
        H=h,
        W=w,
        E=e,
        S0N=int(x0.stride(0)),
        S0C=int(x0.stride(1)),
        S0H=int(x0.stride(2)),
        S0W=int(x0.stride(3)),
        S1N=int(x1.stride(0)),
        S1C=int(x1.stride(1)),
        S1H=int(x1.stride(2)),
        S1W=int(x1.stride(3)),
        NUM_CHUNKS=num_chunks,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        RAW_STATS=RAW_STATS,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _pair_finalize_stats_kernel[(triton.cdiv(max_stats_c, BLOCK_C), 2)](
        partial,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        mean0,
        invstd0,
        mean1,
        invstd1,
        C0=c0,
        C1=c1,
        TOTAL_C=stats_c,
        E=e,
        NUM_CHUNKS=num_chunks,
        BLOCK_R=BLOCK_R,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_C=BLOCK_C,
        RAW_STATS=RAW_STATS,
        num_warps=num_warps_final,
        num_stages=3,
    )

    branch0_total = n * c0 * h * w
    branch1_total = n * c1 * h * w
    _bn_relu_cat_branch_kernel[(triton.cdiv(branch0_total, CAT_BLOCK),)](
        x0,
        weight0,
        bias0,
        mean0,
        invstd0,
        cat,
        OUT_OFFSET=0,
        C=c0,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=branch0_total,
        SN=int(x0.stride(0)),
        SC=int(x0.stride(1)),
        SH=int(x0.stride(2)),
        SW=int(x0.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_cat_branch_kernel[(triton.cdiv(branch1_total, CAT_BLOCK),)](
        x1,
        weight1,
        bias1,
        mean1,
        invstd1,
        cat,
        OUT_OFFSET=c0,
        C=c1,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=branch1_total,
        SN=int(x1.stride(0)),
        SC=int(x1.stride(1)),
        SH=int(x1.stride(2)),
        SW=int(x1.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _maxpool_cat_kernel[(triton.cdiv(pool_branch_total, POOL_BRANCH_BLOCK),)](
        pool_x,
        cat,
        pool_offsets,
        OUT_OFFSET=stats_c,
        C=pool_c,
        TOTAL_C=total_c,
        H_IN=pool_h_in,
        W_IN=pool_w_in,
        H_OUT=h,
        W_OUT=w,
        TOTAL=pool_branch_total,
        SN=int(pool_x.stride(0)),
        SC=int(pool_x.stride(1)),
        SH=int(pool_x.stride(2)),
        SW=int(pool_x.stride(3)),
        BLOCK=POOL_BRANCH_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _avg_pool3x3_kernel[(triton.cdiv(total, AVG_BLOCK),)](
        cat,
        avg_pool,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=total,
        BLOCK=AVG_BLOCK,
        num_warps=4,
        num_stages=3,
    )

    return (
        mean0,
        invstd0,
        mean1,
        invstd1,
        pool_offsets,
        cat,
        avg_pool,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )


# 5b79c9f7: (T([128,320,8,8], bf16), T([128,192,8,8], bf16), T([128,768,17,17], bf16))
@oracle_impl(hardware="B200", point="5b79c9f7", BLOCK_R=1024, BLOCK_C=8, CAT_BLOCK=1024, POOL_BRANCH_BLOCK=512, AVG_BLOCK=512, num_warps_stats=8, num_warps_final=4, RAW_STATS=True)
# 505c1a16: (T([128,384,17,17], bf16), T([128,96,17,17], bf16), T([128,288,35,35], bf16))
@oracle_impl(hardware="B200", point="505c1a16", BLOCK_R=1024, BLOCK_C=8, CAT_BLOCK=1024, POOL_BRANCH_BLOCK=512, AVG_BLOCK=512, num_warps_stats=8, num_warps_final=4, RAW_STATS=True)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    CAT_BLOCK: int,
    POOL_BRANCH_BLOCK: int,
    AVG_BLOCK: int,
    num_warps_stats: int,
    num_warps_final: int,
    RAW_STATS: bool,
):
    return _run(
        inputs,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        CAT_BLOCK=CAT_BLOCK,
        POOL_BRANCH_BLOCK=POOL_BRANCH_BLOCK,
        AVG_BLOCK=AVG_BLOCK,
        num_warps_stats=num_warps_stats,
        num_warps_final=num_warps_final,
        RAW_STATS=RAW_STATS,
    )
