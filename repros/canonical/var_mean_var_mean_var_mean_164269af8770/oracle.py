"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete six-branch bf16 Inception training-BatchNorm fanout, including each fp32 population `var_mean(..., correction=0)` over channels-last bf16 inputs, eps=0.001 rsqrt side outputs, twelve mutable running-stat `copy_` updates with the captured 8192/8191 correction literal, exact fp32 affine followed by explicit bf16 cast and bf16 ReLU, the returned channels-last bf16 `[128,2048,8,8]` channel cat, and the returned padded 3x3 avg_pool2d tail, whereas Inductor lowers the sibling BN-training branches, mutable copy_ epilogues, nested channel cats, and pooling consumer as separate generic reduction/pointwise/cat/pool schedules; Inductor cannot do this today because the scheduler does not co-schedule multiple independent training-BN templates with mutable aliases while sinking the static channel-cat layout and pool tail into one full-scope plan; the fix is SCHEDULER_FUSION: extend the training-BatchNorm scheduler to group sibling channel-stat reductions, emit running-stat side effects, and fuse exact bf16 cat/pool materialization for static Inception branch fanouts."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

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
def _six_partial_stats_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    x4_ptr,
    x5_ptr,
    partial_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    C5: tl.constexpr,
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
    S2N: tl.constexpr,
    S2C: tl.constexpr,
    S2H: tl.constexpr,
    S2W: tl.constexpr,
    S3N: tl.constexpr,
    S3C: tl.constexpr,
    S3H: tl.constexpr,
    S3W: tl.constexpr,
    S4N: tl.constexpr,
    S4C: tl.constexpr,
    S4H: tl.constexpr,
    S4W: tl.constexpr,
    S5N: tl.constexpr,
    S5C: tl.constexpr,
    S5H: tl.constexpr,
    S5W: tl.constexpr,
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
    is2 = branch == 2
    is3 = branch == 3
    is4 = branch == 4
    is5 = branch == 5
    live0 = is0 & (channels < C0)
    live1 = is1 & (channels < C1)
    live2 = is2 & (channels < C2)
    live3 = is3 & (channels < C3)
    live4 = is4 & (channels < C4)
    live5 = is5 & (channels < C5)
    channel_live = live0 | live1 | live2 | live3 | live4 | live5
    active = channel_live[:, None] & r_live[None, :]

    off0 = n_idx[None, :] * S0N + channels[:, None] * S0C + h_idx[None, :] * S0H + w_idx[None, :] * S0W
    off1 = n_idx[None, :] * S1N + channels[:, None] * S1C + h_idx[None, :] * S1H + w_idx[None, :] * S1W
    off2 = n_idx[None, :] * S2N + channels[:, None] * S2C + h_idx[None, :] * S2H + w_idx[None, :] * S2W
    off3 = n_idx[None, :] * S3N + channels[:, None] * S3C + h_idx[None, :] * S3H + w_idx[None, :] * S3W
    off4 = n_idx[None, :] * S4N + channels[:, None] * S4C + h_idx[None, :] * S4H + w_idx[None, :] * S4W
    off5 = n_idx[None, :] * S5N + channels[:, None] * S5C + h_idx[None, :] * S5H + w_idx[None, :] * S5W

    vals0 = tl.load(x0_ptr + off0, mask=live0[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals1 = tl.load(x1_ptr + off1, mask=live1[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals2 = tl.load(x2_ptr + off2, mask=live2[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals3 = tl.load(x3_ptr + off3, mask=live3[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals4 = tl.load(x4_ptr + off4, mask=live4[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals5 = tl.load(x5_ptr + off5, mask=live5[:, None] & r_live[None, :], other=0.0).to(tl.float32)
    vals = _f32_add(_f32_add(vals0, vals1), _f32_add(vals2, vals3))
    vals = tl.where(active, _f32_add(vals, _f32_add(vals4, vals5)), 0.0)

    if RAW_STATS:
        partial0 = tl.sum(vals, axis=1)
        partial1 = tl.sum(_f32_mul(vals, vals), axis=1)
    else:
        weights = tl.where(active, 1.0, 0.0)
        zero = tl.zeros((BLOCK_C, BLOCK_R), tl.float32)
        partial0, partial1, _ = triton_helpers.welford(vals, zero, weights, 1)

    global_channels = tl.where(
        is0,
        channels,
        tl.where(
            is1,
            C0 + channels,
            tl.where(
                is2,
                C0 + C1 + channels,
                tl.where(
                    is3,
                    C0 + C1 + C2 + channels,
                    tl.where(is4, C0 + C1 + C2 + C3 + channels, C0 + C1 + C2 + C3 + C4 + channels),
                ),
            ),
        ),
    )
    plane = NUM_CHUNKS * TOTAL_C
    store_offsets = chunk * TOTAL_C + global_channels
    tl.store(partial_ptr + store_offsets, partial0, mask=channel_live)
    tl.store(partial_ptr + plane + store_offsets, partial1, mask=channel_live)


@triton.jit
def _six_finalize_stats_kernel(
    partial_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    running_mean2_ptr,
    running_var2_ptr,
    running_mean3_ptr,
    running_var3_ptr,
    running_mean4_ptr,
    running_var4_ptr,
    running_mean5_ptr,
    running_var5_ptr,
    mean0_ptr,
    invstd0_ptr,
    mean1_ptr,
    invstd1_ptr,
    mean2_ptr,
    invstd2_ptr,
    mean3_ptr,
    invstd3_ptr,
    mean4_ptr,
    invstd4_ptr,
    mean5_ptr,
    invstd5_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    C5: tl.constexpr,
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
    is2 = branch == 2
    is3 = branch == 3
    is4 = branch == 4
    is5 = branch == 5
    live0 = is0 & (channels < C0)
    live1 = is1 & (channels < C1)
    live2 = is2 & (channels < C2)
    live3 = is3 & (channels < C3)
    live4 = is4 & (channels < C4)
    live5 = is5 & (channels < C5)
    channel_live = live0 | live1 | live2 | live3 | live4 | live5
    global_channels = tl.where(
        is0,
        channels,
        tl.where(
            is1,
            C0 + channels,
            tl.where(
                is2,
                C0 + C1 + channels,
                tl.where(
                    is3,
                    C0 + C1 + C2 + channels,
                    tl.where(is4, C0 + C1 + C2 + C3 + channels, C0 + C1 + C2 + C3 + C4 + channels),
                ),
            ),
        ),
    )

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

    old_mean0 = tl.load(running_mean0_ptr + channels, mask=live0, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + channels, mask=live0, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + channels, mask=live1, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + channels, mask=live1, other=0.0).to(tl.float32)
    old_mean2 = tl.load(running_mean2_ptr + channels, mask=live2, other=0.0).to(tl.float32)
    old_var2 = tl.load(running_var2_ptr + channels, mask=live2, other=0.0).to(tl.float32)
    old_mean3 = tl.load(running_mean3_ptr + channels, mask=live3, other=0.0).to(tl.float32)
    old_var3 = tl.load(running_var3_ptr + channels, mask=live3, other=0.0).to(tl.float32)
    old_mean4 = tl.load(running_mean4_ptr + channels, mask=live4, other=0.0).to(tl.float32)
    old_var4 = tl.load(running_var4_ptr + channels, mask=live4, other=0.0).to(tl.float32)
    old_mean5 = tl.load(running_mean5_ptr + channels, mask=live5, other=0.0).to(tl.float32)
    old_var5 = tl.load(running_var5_ptr + channels, mask=live5, other=0.0).to(tl.float32)

    running_var_update = _f32_mul(_f32_mul(var, 1.0001220852154804), 0.1)
    new_mean0 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean0, 0.9))
    new_var0 = _f32_add(running_var_update, _f32_mul(old_var0, 0.9))
    new_mean1 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean1, 0.9))
    new_var1 = _f32_add(running_var_update, _f32_mul(old_var1, 0.9))
    new_mean2 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean2, 0.9))
    new_var2 = _f32_add(running_var_update, _f32_mul(old_var2, 0.9))
    new_mean3 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean3, 0.9))
    new_var3 = _f32_add(running_var_update, _f32_mul(old_var3, 0.9))
    new_mean4 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean4, 0.9))
    new_var4 = _f32_add(running_var_update, _f32_mul(old_var4, 0.9))
    new_mean5 = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean5, 0.9))
    new_var5 = _f32_add(running_var_update, _f32_mul(old_var5, 0.9))

    tl.store(mean0_ptr + channels, mean, mask=live0)
    tl.store(invstd0_ptr + channels, invstd, mask=live0)
    tl.store(running_mean0_ptr + channels, new_mean0, mask=live0)
    tl.store(running_var0_ptr + channels, new_var0, mask=live0)

    tl.store(mean1_ptr + channels, mean, mask=live1)
    tl.store(invstd1_ptr + channels, invstd, mask=live1)
    tl.store(running_mean1_ptr + channels, new_mean1, mask=live1)
    tl.store(running_var1_ptr + channels, new_var1, mask=live1)

    tl.store(mean2_ptr + channels, mean, mask=live2)
    tl.store(invstd2_ptr + channels, invstd, mask=live2)
    tl.store(running_mean2_ptr + channels, new_mean2, mask=live2)
    tl.store(running_var2_ptr + channels, new_var2, mask=live2)

    tl.store(mean3_ptr + channels, mean, mask=live3)
    tl.store(invstd3_ptr + channels, invstd, mask=live3)
    tl.store(running_mean3_ptr + channels, new_mean3, mask=live3)
    tl.store(running_var3_ptr + channels, new_var3, mask=live3)

    tl.store(mean4_ptr + channels, mean, mask=live4)
    tl.store(invstd4_ptr + channels, invstd, mask=live4)
    tl.store(running_mean4_ptr + channels, new_mean4, mask=live4)
    tl.store(running_var4_ptr + channels, new_var4, mask=live4)

    tl.store(mean5_ptr + channels, mean, mask=live5)
    tl.store(invstd5_ptr + channels, invstd, mask=live5)
    tl.store(running_mean5_ptr + channels, new_mean5, mask=live5)
    tl.store(running_var5_ptr + channels, new_var5, mask=live5)


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
    POOL_BLOCK: int,
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
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
        x3,
        running_mean3,
        running_var3,
        weight3,
        bias3,
        x4,
        running_mean4,
        running_var4,
        weight4,
        bias4,
        x5,
        running_mean5,
        running_var5,
        weight5,
        bias5,
    ) = inputs

    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    c2 = int(x2.shape[1])
    c3 = int(x3.shape[1])
    c4 = int(x4.shape[1])
    c5 = int(x5.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    e = n * h * w
    total_c = c0 + c1 + c2 + c3 + c4 + c5
    max_c = max(c0, c1, c2, c3, c4, c5)
    total = n * total_c * h * w
    num_chunks = triton.cdiv(e, BLOCK_R)
    block_chunks = _next_power_of_2(num_chunks)

    partial = torch.empty((2, num_chunks, total_c), device=x0.device, dtype=torch.float32)
    mean0 = torch.empty_strided((1, c0, 1, 1), (c0, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd0 = torch.empty_strided((1, c0, 1, 1), (c0, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, c1, 1, 1), (c1, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty_strided((1, c1, 1, 1), (c1, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean2 = torch.empty_strided((1, c2, 1, 1), (c2, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd2 = torch.empty_strided((1, c2, 1, 1), (c2, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean3 = torch.empty_strided((1, c3, 1, 1), (c3, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd3 = torch.empty_strided((1, c3, 1, 1), (c3, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean4 = torch.empty_strided((1, c4, 1, 1), (c4, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd4 = torch.empty_strided((1, c4, 1, 1), (c4, 1, 1, 1), device=x0.device, dtype=torch.float32)
    mean5 = torch.empty_strided((1, c5, 1, 1), (c5, 1, 1, 1), device=x0.device, dtype=torch.float32)
    invstd5 = torch.empty_strided((1, c5, 1, 1), (c5, 1, 1, 1), device=x0.device, dtype=torch.float32)
    cat = torch.empty_strided(
        (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    pool = torch.empty_strided(
        (n, total_c, h, w),
        (total_c * h * w, 1, total_c * w, total_c),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    _six_partial_stats_kernel[(triton.cdiv(max_c, BLOCK_C), num_chunks, 6)](
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        partial,
        C0=c0,
        C1=c1,
        C2=c2,
        C3=c3,
        C4=c4,
        C5=c5,
        TOTAL_C=total_c,
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
        S2N=int(x2.stride(0)),
        S2C=int(x2.stride(1)),
        S2H=int(x2.stride(2)),
        S2W=int(x2.stride(3)),
        S3N=int(x3.stride(0)),
        S3C=int(x3.stride(1)),
        S3H=int(x3.stride(2)),
        S3W=int(x3.stride(3)),
        S4N=int(x4.stride(0)),
        S4C=int(x4.stride(1)),
        S4H=int(x4.stride(2)),
        S4W=int(x4.stride(3)),
        S5N=int(x5.stride(0)),
        S5C=int(x5.stride(1)),
        S5H=int(x5.stride(2)),
        S5W=int(x5.stride(3)),
        NUM_CHUNKS=num_chunks,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        RAW_STATS=RAW_STATS,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _six_finalize_stats_kernel[(triton.cdiv(max_c, BLOCK_C), 6)](
        partial,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        running_mean3,
        running_var3,
        running_mean4,
        running_var4,
        running_mean5,
        running_var5,
        mean0,
        invstd0,
        mean1,
        invstd1,
        mean2,
        invstd2,
        mean3,
        invstd3,
        mean4,
        invstd4,
        mean5,
        invstd5,
        C0=c0,
        C1=c1,
        C2=c2,
        C3=c3,
        C4=c4,
        C5=c5,
        TOTAL_C=total_c,
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
    branch2_total = n * c2 * h * w
    branch3_total = n * c3 * h * w
    branch4_total = n * c4 * h * w
    branch5_total = n * c5 * h * w
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
    _bn_relu_cat_branch_kernel[(triton.cdiv(branch2_total, CAT_BLOCK),)](
        x2,
        weight2,
        bias2,
        mean2,
        invstd2,
        cat,
        OUT_OFFSET=c0 + c1,
        C=c2,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=branch2_total,
        SN=int(x2.stride(0)),
        SC=int(x2.stride(1)),
        SH=int(x2.stride(2)),
        SW=int(x2.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_cat_branch_kernel[(triton.cdiv(branch3_total, CAT_BLOCK),)](
        x3,
        weight3,
        bias3,
        mean3,
        invstd3,
        cat,
        OUT_OFFSET=c0 + c1 + c2,
        C=c3,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=branch3_total,
        SN=int(x3.stride(0)),
        SC=int(x3.stride(1)),
        SH=int(x3.stride(2)),
        SW=int(x3.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_cat_branch_kernel[(triton.cdiv(branch4_total, CAT_BLOCK),)](
        x4,
        weight4,
        bias4,
        mean4,
        invstd4,
        cat,
        OUT_OFFSET=c0 + c1 + c2 + c3,
        C=c4,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=branch4_total,
        SN=int(x4.stride(0)),
        SC=int(x4.stride(1)),
        SH=int(x4.stride(2)),
        SW=int(x4.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_cat_branch_kernel[(triton.cdiv(branch5_total, CAT_BLOCK),)](
        x5,
        weight5,
        bias5,
        mean5,
        invstd5,
        cat,
        OUT_OFFSET=c0 + c1 + c2 + c3 + c4,
        C=c5,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=branch5_total,
        SN=int(x5.stride(0)),
        SC=int(x5.stride(1)),
        SH=int(x5.stride(2)),
        SW=int(x5.stride(3)),
        BLOCK=CAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _avg_pool3x3_kernel[(triton.cdiv(total, POOL_BLOCK),)](
        cat,
        pool,
        TOTAL_C=total_c,
        H=h,
        W=w,
        TOTAL=total,
        BLOCK=POOL_BLOCK,
        num_warps=4,
        num_stages=3,
    )

    return (
        mean0,
        invstd0,
        mean1,
        invstd1,
        mean2,
        invstd2,
        mean3,
        invstd3,
        mean4,
        invstd4,
        mean5,
        invstd5,
        cat,
        pool,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        running_mean3,
        running_var3,
        running_mean4,
        running_var4,
        running_mean5,
        running_var5,
    )


# 13e49e96: six channels-last bf16 Inception training-BN branches -> cat + avg_pool.
@oracle_impl(hardware="B200", point="13e49e96", BLOCK_R=1024, BLOCK_C=8, CAT_BLOCK=1024, POOL_BLOCK=512, num_warps_stats=8, num_warps_final=4, RAW_STATS=True)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    CAT_BLOCK: int,
    POOL_BLOCK: int,
    num_warps_stats: int,
    num_warps_final: int,
    RAW_STATS: bool,
):
    return _run(
        inputs,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        CAT_BLOCK=CAT_BLOCK,
        POOL_BLOCK=POOL_BLOCK,
        num_warps_stats=num_warps_stats,
        num_warps_final=num_warps_final,
        RAW_STATS=RAW_STATS,
    )
