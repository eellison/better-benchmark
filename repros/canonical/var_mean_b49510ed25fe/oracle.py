"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete PyTorch-UNet bf16 training-BatchNorm, running-stat copy_ updates, affine bf16-cast ReLU, returned mean/rsqrt/ReLU side outputs, and low-memory 2x2 stride-2 maxpool-with-offsets scope with tiled channel reductions and direct bf16 pooling, whereas Inductor lowers the channel var_mean, mutable running-stat epilogues, affine/ReLU materialization, and maxpool consumer as generic separated schedules; Inductor cannot do this today because scheduler fusion does not keep a reduction producer with mutable side effects and returned normalization tensors live across a multi-output low-memory maxpool consumer while preserving the bf16 cast-before-ReLU boundary and int8 offset semantics; the fix is SCHEDULER_FUSION: teach the scheduler a training-BatchNorm-affine-ReLU-to-low-memory-maxpool lowering that emits returned stats, ReLU, pool values/offsets, and running-stat copy_ epilogues in one full-scope plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1


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
def _bn_direct_stats_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rsqrt_ptr,
    running_mean_ptr,
    running_var_ptr,
    relu_ptr,
    HW: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_M)
    active = offsets < HW
    x = tl.load(
        x_ptr + channel * HW + offsets,
        mask=active,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    mean_acc = tl.zeros([BLOCK_M], tl.float32)
    m2_acc = tl.zeros([BLOCK_M], tl.float32)
    weight_acc = tl.zeros([BLOCK_M], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(active, mean_next, mean_acc)
    m2_acc = tl.where(active, m2_next, m2_acc)
    weight_acc = tl.where(active, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)
    var = m2 / HW
    inv_std = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(mean_ptr + channel, mean)
    tl.store(rsqrt_ptr + channel, inv_std)
    tl.store(running_mean_ptr + channel, _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9)))
    tl.store(
        running_var_ptr + channel,
        _f32_add(_f32_mul(_f32_mul(var, RUNNING_VAR_CORRECTION), 0.1), _f32_mul(old_var, 0.9)),
    )

    affine_weight = tl.load(weight_ptr + channel).to(tl.float32)
    affine_bias = tl.load(bias_ptr + channel).to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, inv_std)
    scaled = _f32_mul(normalized, affine_weight)
    affine_bf16 = _f32_add(scaled, affine_bias).to(tl.bfloat16)
    relu = tl.where((affine_bf16 > 0.0) | (affine_bf16 != affine_bf16), affine_bf16, 0.0)
    tl.store(relu_ptr + channel * HW + offsets, relu, mask=active)


@triton.jit
def _bn_direct_stats_kernel(
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    running_mean_ptr,
    running_var_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_M)
    active = offsets < HW
    x = tl.load(
        x_ptr + channel * HW + offsets,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    mean_acc = tl.zeros([BLOCK_M], tl.float32)
    m2_acc = tl.zeros([BLOCK_M], tl.float32)
    weight_acc = tl.zeros([BLOCK_M], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(active, mean_next, mean_acc)
    m2_acc = tl.where(active, m2_next, m2_acc)
    weight_acc = tl.where(active, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 0)
    var = m2 / HW
    inv_std = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(mean_ptr + channel, mean)
    tl.store(rsqrt_ptr + channel, inv_std)
    tl.store(running_mean_ptr + channel, _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9)))
    tl.store(
        running_var_ptr + channel,
        _f32_add(_f32_mul(_f32_mul(var, RUNNING_VAR_CORRECTION), 0.1), _f32_mul(old_var, 0.9)),
    )


@triton.jit
def _bn_partial_stats_kernel(
    x_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK_M: tl.constexpr,
):
    channel = tl.program_id(0)
    partial_id = tl.program_id(1)
    chunk_offsets = tl.arange(0, BLOCK_M)
    offsets = partial_id * CHUNK_M + chunk_offsets
    active = (chunk_offsets < CHUNK_M) & (offsets < HW)
    x = tl.load(
        x_ptr + channel * HW + offsets,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    mean_acc = tl.zeros([BLOCK_M], tl.float32)
    m2_acc = tl.zeros([BLOCK_M], tl.float32)
    weight_acc = tl.zeros([BLOCK_M], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(active, mean_next, mean_acc)
    m2_acc = tl.where(active, m2_next, m2_acc)
    weight_acc = tl.where(active, weight_next, weight_acc)
    block_mean, block_m2, block_weight = triton_helpers.welford(
        mean_acc,
        m2_acc,
        weight_acc,
        0,
    )
    partial_offset = channel * NUM_PARTIALS + partial_id
    tl.store(partial_mean_ptr + partial_offset, block_mean)
    tl.store(partial_m2_ptr + partial_offset, block_m2)
    tl.store(partial_weight_ptr + partial_offset, block_weight)


@triton.jit
def _bn_partial_stats_x_kernel(
    x_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK_M: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[:, None]
    xmask = xindex < (C * NUM_PARTIALS)
    r_base = tl.arange(0, BLOCK_M)[None, :]
    channel = xindex // NUM_PARTIALS
    partial_id = xindex - channel * NUM_PARTIALS
    mean_acc = tl.zeros([X_BLOCK, BLOCK_M], tl.float32)
    m2_acc = tl.zeros([X_BLOCK, BLOCK_M], tl.float32)
    weight_acc = tl.zeros([X_BLOCK, BLOCK_M], tl.float32)
    for r_offset in range(0, CHUNK_M, BLOCK_M):
        r_offsets = r_offset + r_base
        active = xmask & (r_offsets < CHUNK_M)
        offsets = channel * HW + partial_id * CHUNK_M + r_offsets
        x = tl.load(
            x_ptr + offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            x,
            mean_acc,
            m2_acc,
            weight_acc,
            r_offset == 0,
        )
        mean_acc = tl.where(active, mean_next, mean_acc)
        m2_acc = tl.where(active, m2_next, m2_acc)
        weight_acc = tl.where(active, weight_next, weight_acc)
    block_mean, block_m2, block_weight = triton_helpers.welford(
        mean_acc,
        m2_acc,
        weight_acc,
        1,
    )
    partial_offset = (tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK))
    store_mask = partial_offset < (C * NUM_PARTIALS)
    tl.store(partial_mean_ptr + partial_offset, block_mean, mask=store_mask)
    tl.store(partial_m2_ptr + partial_offset, block_m2, mask=store_mask)
    tl.store(partial_weight_ptr + partial_offset, block_weight, mask=store_mask)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    mean_ptr,
    rsqrt_ptr,
    running_mean_ptr,
    running_var_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    active = offsets < NUM_PARTIALS
    partial_base = channel * NUM_PARTIALS + offsets

    block_mean = tl.load(partial_mean_ptr + partial_base, mask=active, other=0.0).to(tl.float32)
    block_m2 = tl.load(partial_m2_ptr + partial_base, mask=active, other=0.0).to(tl.float32)
    block_weight = tl.load(partial_weight_ptr + partial_base, mask=active, other=0.0).to(tl.float32)
    mean_inputs = tl.where(active, block_mean, 0.0)
    m2_inputs = tl.where(active, block_m2, 0.0)
    weight_inputs = tl.where(active, block_weight, 0.0)
    mean, m2, _weight = triton_helpers.welford(mean_inputs, m2_inputs, weight_inputs, 0)
    var = m2 / HW
    inv_std = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(mean_ptr + channel, mean)
    tl.store(rsqrt_ptr + channel, inv_std)
    tl.store(running_mean_ptr + channel, _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9)))
    tl.store(
        running_var_ptr + channel,
        _f32_add(_f32_mul(_f32_mul(var, RUNNING_VAR_CORRECTION), 0.1), _f32_mul(old_var, 0.9)),
    )


@triton.jit
def _bn_finalize_stats_x_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    mean_ptr,
    rsqrt_ptr,
    running_mean_ptr,
    running_var_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
    X_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * X_BLOCK
    channels = xoffset + tl.arange(0, X_BLOCK)[:, None]
    xmask = channels < C
    r_offsets = tl.arange(0, R_BLOCK)[None, :]
    rmask = r_offsets < NUM_PARTIALS
    partial_base = channels * NUM_PARTIALS + r_offsets

    block_mean = tl.load(partial_mean_ptr + partial_base, mask=xmask & rmask, other=0.0).to(tl.float32)
    block_m2 = tl.load(partial_m2_ptr + partial_base, mask=xmask & rmask, other=0.0).to(tl.float32)
    block_weight = tl.load(partial_weight_ptr + partial_base, mask=xmask & rmask, other=0.0).to(tl.float32)
    mean_inputs = tl.where(xmask & rmask, block_mean, 0.0)
    m2_inputs = tl.where(xmask & rmask, block_m2, 0.0)
    weight_inputs = tl.where(xmask & rmask, block_weight, 0.0)
    mean, m2, _weight = triton_helpers.welford(mean_inputs, m2_inputs, weight_inputs, 1)
    channel_1d = xoffset + tl.arange(0, X_BLOCK)
    active = channel_1d < C
    var = m2 / HW
    inv_std = libdevice.rsqrt(var + 1.0e-5)

    old_mean = tl.load(running_mean_ptr + channel_1d, mask=active, eviction_policy="evict_last").to(tl.float32)
    old_var = tl.load(running_var_ptr + channel_1d, mask=active, eviction_policy="evict_last").to(tl.float32)
    tl.store(mean_ptr + channel_1d, mean, mask=active)
    tl.store(rsqrt_ptr + channel_1d, inv_std, mask=active)
    tl.store(running_mean_ptr + channel_1d, mean * 0.1 + old_mean * 0.9, mask=active)
    tl.store(
        running_var_ptr + channel_1d,
        (var * RUNNING_VAR_CORRECTION) * 0.1 + old_var * 0.9,
        mask=active,
    )


@triton.jit
def _bn_affine_relu_pool2x2_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rsqrt_ptr,
    relu_ptr,
    values_ptr,
    offsets_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUT_W: tl.constexpr,
    HW: tl.constexpr,
    OUT_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    SEPARATE_AFFINE: tl.constexpr,
    TIE_LAST: tl.constexpr,
    STORE_TAIL: tl.constexpr,
):
    c_block = tl.program_id(0)
    out_block = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_active = out_offsets < OUT_HW
    c_active = channels < C
    out_h = out_offsets // OUT_W
    out_w = out_offsets - out_h * OUT_W

    mean = tl.load(mean_ptr + channels, mask=c_active, other=0.0).to(tl.float32)
    inv_std = tl.load(rsqrt_ptr + channels, mask=c_active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=c_active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=c_active, other=0.0).to(tl.float32)
    in_base = channels[:, None] * HW
    out_base = channels[:, None] * OUT_HW + out_offsets[None, :]
    best = tl.full((BLOCK_C, BLOCK_OUT), -float("inf"), tl.float32)
    best_offset = tl.zeros((BLOCK_C, BLOCK_OUT), tl.int32)

    for kh in tl.static_range(0, 2):
        in_h = out_h * 2 + kh
        for kw in tl.static_range(0, 2):
            in_w = out_w * 2 + kw
            valid = c_active[:, None] & out_active[None, :] & (in_h[None, :] < H) & (in_w[None, :] < W)
            in_offsets = in_base + in_h[None, :] * W + in_w[None, :]
            x = tl.load(
                x_ptr + in_offsets,
                mask=valid,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            if SEPARATE_AFFINE:
                affine = ((x - mean[:, None]) * inv_std[:, None]) * weight[:, None] + bias[:, None]
            else:
                affine = ((x - mean[:, None]) * inv_std[:, None]) * weight[:, None] + bias[:, None]
            relu = triton_helpers.maximum(tl.full(affine.shape, 0, tl.int32), affine.to(tl.float32))
            tl.store(relu_ptr + in_offsets, relu, mask=valid)
            value = relu.to(tl.float32)
            if TIE_LAST:
                take = valid & ((value > best) | (value != value) | ((value == best) & (value != 0.0)))
            else:
                take = valid & ((value > best) | (value != value))
            best = tl.where(take, value, best)
            best_offset = tl.where(take, kh * 2 + kw, best_offset)

    store_mask = c_active[:, None] & out_active[None, :]
    tl.store(values_ptr + out_base, best, mask=store_mask)
    tl.store(offsets_ptr + out_base, best_offset.to(tl.int8), mask=store_mask)

    if STORE_TAIL:
        tail_rows = out_offsets
        tail_valid = c_active[:, None] & (out_block == 0) & (tail_rows[None, :] < H)
        tail_offsets = in_base + tail_rows[None, :] * W + (W - 1)
        x_tail = tl.load(
            x_ptr + tail_offsets,
            mask=tail_valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        affine_tail = ((x_tail - mean[:, None]) * inv_std[:, None]) * weight[:, None] + bias[:, None]
        relu_tail = triton_helpers.maximum(tl.full(affine_tail.shape, 0, tl.int32), affine_tail.to(tl.float32))
        tl.store(relu_ptr + tail_offsets, relu_tail, mask=tail_valid)


@triton.jit
def _bn_affine_relu_tail_col_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rsqrt_ptr,
    relu_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
    SEPARATE_AFFINE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = C * H
    active = offsets < total
    channel = offsets // H
    row = offsets - channel * H
    in_offsets = channel * HW + row * W + (W - 1)

    x = tl.load(x_ptr + in_offsets, mask=active, other=0.0, eviction_policy="evict_first").to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    inv_std = tl.load(rsqrt_ptr + channel, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=active, other=0.0).to(tl.float32)
    if SEPARATE_AFFINE:
        affine = ((x - mean) * inv_std) * weight + bias
    else:
        affine = ((x - mean) * inv_std) * weight + bias
    relu = triton_helpers.maximum(tl.full(affine.shape, 0, tl.int32), affine.to(tl.float32))
    tl.store(relu_ptr + in_offsets, relu, mask=active)


@triton.jit
def _bn_affine_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rsqrt_ptr,
    relu_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
    SEPARATE_AFFINE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    channel = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=active, other=0.0, eviction_policy="evict_first").to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    inv_std = tl.load(rsqrt_ptr + channel, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=active, other=0.0).to(tl.float32)
    if SEPARATE_AFFINE:
        affine = ((x - mean) * inv_std) * weight + bias
    else:
        affine = ((x - mean) * inv_std) * weight + bias
    relu = triton_helpers.maximum(tl.full(affine.shape, 0, tl.int32), affine.to(tl.float32))
    tl.store(relu_ptr + offsets, relu, mask=active)


@triton.jit
def _maxpool2x2_kernel(
    relu_ptr,
    values_ptr,
    offsets_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    HW: tl.constexpr,
    OUT_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    TIE_LAST: tl.constexpr,
):
    c_block = tl.program_id(0)
    out_block = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_active = out_offsets < OUT_HW
    c_active = channels < C
    out_h = out_offsets // OUT_W
    out_w = out_offsets - out_h * OUT_W

    in_base = channels[:, None] * HW
    out_base = channels[:, None] * OUT_HW + out_offsets[None, :]
    best = tl.full((BLOCK_C, BLOCK_OUT), -float("inf"), tl.float32)
    best_offset = tl.zeros((BLOCK_C, BLOCK_OUT), tl.int32)

    for kh in tl.static_range(0, 2):
        in_h = out_h * 2 + kh
        for kw in tl.static_range(0, 2):
            in_w = out_w * 2 + kw
            valid = c_active[:, None] & out_active[None, :] & (in_h[None, :] < H) & (in_w[None, :] < W)
            value = tl.load(
                relu_ptr + in_base + in_h[None, :] * W + in_w[None, :],
                mask=valid,
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            if TIE_LAST:
                take = valid & ((value > best) | (value != value) | ((value == best) & (value != 0.0)))
            else:
                take = valid & ((value > best) | (value != value))
            best = tl.where(take, value, best)
            best_offset = tl.where(take, kh * 2 + kw, best_offset)

    store_mask = c_active[:, None] & out_active[None, :]
    tl.store(values_ptr + out_base, best, mask=store_mask)
    tl.store(offsets_ptr + out_base, best_offset.to(tl.int8), mask=store_mask)


@oracle_impl(hardware="B200", point="6750c5ad", STAT_BLOCK=16384, STAT_CHUNK=16384, VECTOR_PARTIAL=False, PARTIAL_XBLOCK=1, VECTOR_FINAL=False, DIRECT_STATS=True, DIRECT_RELU=True, SEPARATE_AFFINE=False, FUSED_POOL=False, TIE_LAST=False, STORE_TAIL=False, RELU_BLOCK=256, POOL_BLOCK_C=1, POOL_BLOCK_OUT=128, num_warps_stats=8, num_warps_relu=4, num_warps_pool=4)
@oracle_impl(hardware="B200", point="8e0806dd", STAT_BLOCK=2048, STAT_CHUNK=2048, VECTOR_PARTIAL=False, PARTIAL_XBLOCK=1, VECTOR_FINAL=False, DIRECT_STATS=False, DIRECT_RELU=False, SEPARATE_AFFINE=False, FUSED_POOL=False, TIE_LAST=False, STORE_TAIL=False, RELU_BLOCK=256, POOL_BLOCK_C=1, POOL_BLOCK_OUT=128, num_warps_stats=8, num_warps_relu=4, num_warps_pool=4)
@oracle_impl(hardware="B200", point="538399db", STAT_BLOCK=32768, STAT_CHUNK=30656, VECTOR_PARTIAL=True, PARTIAL_XBLOCK=1, VECTOR_FINAL=True, DIRECT_STATS=False, DIRECT_RELU=False, SEPARATE_AFFINE=False, FUSED_POOL=False, TIE_LAST=False, STORE_TAIL=False, RELU_BLOCK=256, POOL_BLOCK_C=1, POOL_BLOCK_OUT=128, num_warps_stats=8, num_warps_relu=4, num_warps_pool=4)
@oracle_impl(hardware="B200", point="37819cd3", STAT_BLOCK=2048, STAT_CHUNK=2048, VECTOR_PARTIAL=False, PARTIAL_XBLOCK=1, VECTOR_FINAL=False, DIRECT_STATS=False, DIRECT_RELU=False, SEPARATE_AFFINE=False, FUSED_POOL=False, TIE_LAST=False, STORE_TAIL=False, RELU_BLOCK=256, POOL_BLOCK_C=1, POOL_BLOCK_OUT=128, num_warps_stats=8, num_warps_relu=4, num_warps_pool=4)
def oracle_forward(
    inputs,
    *,
    STAT_BLOCK: int,
    STAT_CHUNK: int,
    VECTOR_PARTIAL: bool,
    PARTIAL_XBLOCK: int,
    VECTOR_FINAL: bool,
    DIRECT_STATS: bool,
    DIRECT_RELU: bool,
    SEPARATE_AFFINE: bool,
    FUSED_POOL: bool,
    TIE_LAST: bool,
    STORE_TAIL: bool,
    RELU_BLOCK: int,
    POOL_BLOCK_C: int,
    POOL_BLOCK_OUT: int,
    num_warps_stats: int,
    num_warps_relu: int,
    num_warps_pool: int,
):
    x, running_mean, running_var, weight, bias, _, _ = inputs
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    total = c * hw
    out_h = h // 2
    out_w = w // 2
    out_hw = out_h * out_w
    num_partials = triton.cdiv(hw, STAT_CHUNK)
    final_block = triton.next_power_of_2(num_partials)
    running_var_correction = 1.0001050530517912

    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    rsqrt = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    relu = torch.empty_strided((1, c, h, w), (total, hw, w, 1), device=x.device, dtype=torch.bfloat16)
    values = torch.empty_strided((1, c, out_h, out_w), (c * out_hw, out_hw, out_w, 1), device=x.device, dtype=torch.bfloat16)
    offsets = torch.empty_strided((1, c, out_h, out_w), (c * out_hw, out_hw, out_w, 1), device=x.device, dtype=torch.int8)
    if DIRECT_STATS:
        if DIRECT_RELU:
            _bn_direct_stats_relu_kernel[(c,)](
                x,
                weight,
                bias,
                mean,
                rsqrt,
                running_mean,
                running_var,
                relu,
                HW=hw,
                RUNNING_VAR_CORRECTION=running_var_correction,
                BLOCK_M=STAT_BLOCK,
                num_warps=num_warps_stats,
                num_stages=4,
            )
        else:
            _bn_direct_stats_kernel[(c,)](
                x,
                mean,
                rsqrt,
                running_mean,
                running_var,
                C=c,
                HW=hw,
                RUNNING_VAR_CORRECTION=running_var_correction,
                BLOCK_M=STAT_BLOCK,
                num_warps=num_warps_stats,
                num_stages=4,
            )
    else:
        partial_mean = torch.empty_strided((c, num_partials), (num_partials, 1), device=x.device, dtype=torch.float32)
        partial_m2 = torch.empty_strided((c, num_partials), (num_partials, 1), device=x.device, dtype=torch.float32)
        partial_weight = torch.empty_strided((c, num_partials), (num_partials, 1), device=x.device, dtype=torch.float32)

        if VECTOR_PARTIAL:
            _bn_partial_stats_x_kernel[(triton.cdiv(c * num_partials, PARTIAL_XBLOCK),)](
                x,
                partial_mean,
                partial_m2,
                partial_weight,
                C=c,
                HW=hw,
                NUM_PARTIALS=num_partials,
                BLOCK_M=STAT_BLOCK,
                CHUNK_M=STAT_CHUNK,
                X_BLOCK=PARTIAL_XBLOCK,
                num_warps=num_warps_stats,
                num_stages=1,
            )
        else:
            _bn_partial_stats_kernel[(c, num_partials)](
                x,
                partial_mean,
                partial_m2,
                partial_weight,
                C=c,
                HW=hw,
                NUM_PARTIALS=num_partials,
                BLOCK_M=STAT_BLOCK,
                CHUNK_M=STAT_CHUNK,
                num_warps=num_warps_stats,
                num_stages=4,
            )
        if VECTOR_FINAL:
            _bn_finalize_stats_x_kernel[(triton.cdiv(c, 32),)](
                partial_mean,
                partial_m2,
                partial_weight,
                mean,
                rsqrt,
                running_mean,
                running_var,
                C=c,
                HW=hw,
                NUM_PARTIALS=num_partials,
                RUNNING_VAR_CORRECTION=running_var_correction,
                X_BLOCK=32,
                R_BLOCK=final_block,
                num_warps=2,
                num_stages=1,
            )
        else:
            _bn_finalize_stats_kernel[(c,)](
                partial_mean,
                partial_m2,
                partial_weight,
                mean,
                rsqrt,
                running_mean,
                running_var,
                C=c,
                HW=hw,
                NUM_PARTIALS=num_partials,
                RUNNING_VAR_CORRECTION=running_var_correction,
                BLOCK_N=final_block,
                num_warps=1,
                num_stages=4,
            )
    if DIRECT_RELU:
        _maxpool2x2_kernel[(triton.cdiv(c, POOL_BLOCK_C), triton.cdiv(out_hw, POOL_BLOCK_OUT))](
            relu,
            values,
            offsets,
            C=c,
            H=h,
            W=w,
            OUT_H=out_h,
            OUT_W=out_w,
            HW=hw,
            OUT_HW=out_hw,
            BLOCK_C=POOL_BLOCK_C,
            BLOCK_OUT=POOL_BLOCK_OUT,
            TIE_LAST=TIE_LAST,
            num_warps=num_warps_pool,
            num_stages=4,
        )
    elif FUSED_POOL:
        _bn_affine_relu_pool2x2_kernel[(triton.cdiv(c, POOL_BLOCK_C), triton.cdiv(out_hw, POOL_BLOCK_OUT))](
            x,
            weight,
            bias,
            mean,
            rsqrt,
            relu,
            values,
            offsets,
            C=c,
            H=h,
            W=w,
            OUT_W=out_w,
            HW=hw,
            OUT_HW=out_hw,
            BLOCK_C=POOL_BLOCK_C,
            BLOCK_OUT=POOL_BLOCK_OUT,
            SEPARATE_AFFINE=SEPARATE_AFFINE,
            TIE_LAST=TIE_LAST,
            STORE_TAIL=STORE_TAIL,
            num_warps=num_warps_pool,
            num_stages=4,
        )
        if not STORE_TAIL:
            _bn_affine_relu_tail_col_kernel[(triton.cdiv(c * h, RELU_BLOCK),)](
                x,
                weight,
                bias,
                mean,
                rsqrt,
                relu,
                C=c,
                H=h,
                W=w,
                HW=hw,
                BLOCK=RELU_BLOCK,
                SEPARATE_AFFINE=SEPARATE_AFFINE,
                num_warps=num_warps_relu,
                num_stages=4,
            )
    else:
        _bn_affine_relu_kernel[(triton.cdiv(total, RELU_BLOCK),)](
            x,
            weight,
            bias,
            mean,
            rsqrt,
            relu,
            TOTAL=total,
            C=c,
            HW=hw,
            BLOCK=RELU_BLOCK,
            SEPARATE_AFFINE=SEPARATE_AFFINE,
            num_warps=num_warps_relu,
            num_stages=4,
        )
        _maxpool2x2_kernel[(triton.cdiv(c, POOL_BLOCK_C), triton.cdiv(out_hw, POOL_BLOCK_OUT))](
            relu,
            values,
            offsets,
            C=c,
            H=h,
            W=w,
            OUT_H=out_h,
            OUT_W=out_w,
            HW=hw,
            OUT_HW=out_hw,
            BLOCK_C=POOL_BLOCK_C,
            BLOCK_OUT=POOL_BLOCK_OUT,
            TIE_LAST=TIE_LAST,
            num_warps=num_warps_pool,
            num_stages=4,
        )
    return mean, rsqrt, relu, values, offsets, running_mean, running_var
