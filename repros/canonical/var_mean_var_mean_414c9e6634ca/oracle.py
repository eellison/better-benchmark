"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 stem with two training-BatchNorm reductions, mutable running-stat copy_ aliases, the first affine/ReLU sunk into the required low-memory 3x3 stride-2 maxpool values and int8 offsets, and the second affine/ReLU output; Inductor lowers the two `var_mean` regions, running-stat epilogues, first activation materialization, maxpool-with-offsets, and second activation as generic schedules with extra producer traffic; the fix is SCHEDULER_FUSION: extend the training-BatchNorm scheduler so a finalized BN statistics plan can feed fixed maxpool stencils and sibling downstream BN reductions while preserving bf16 cast boundaries, side-output aliases, and mutation semantics."""

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
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _bn_partial_stats_kernel(
    x_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_E: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    c_block = tl.program_id(0)
    e_block = tl.program_id(1)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = e_block * BLOCK_E + tl.arange(0, BLOCK_E)
    hw_idx = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    zero = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    weights = tl.where(mask, 1.0, 0.0)
    block_mean, block_m2, block_weight = triton_helpers.welford(
        vals,
        zero,
        weights,
        1,
    )

    out_offsets = e_block * C + channels
    channel_mask = channels < C
    tl.store(partial_mean_ptr + out_offsets, block_mean, mask=channel_mask)
    tl.store(partial_m2_ptr + out_offsets, block_m2, mask=channel_mask)
    tl.store(partial_weight_ptr + out_offsets, block_weight, mask=channel_mask)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    C_BLOCK: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = (channels[:, None] < C) & (chunks[None, :] < NUM_CHUNKS)
    offsets = chunks[None, :] * C + channels[:, None]

    block_mean = tl.load(partial_mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_m2 = tl.load(partial_m2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_weight = tl.load(partial_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean, m2, _weight = triton_helpers.welford(block_mean, block_m2, block_weight, 1)
    var = m2 / E
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    channel_mask = channels < C
    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    updated_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, RUNNING_VAR_CORRECTION)
    updated_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))

    tl.store(mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, updated_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, updated_var, mask=channel_mask)


@triton.jit
def _bn_relu_maxpool3_partial_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    values_ptr,
    offsets_ptr,
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    NUM_OUT_BLOCKS: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    n_idx = tl.program_id(0)
    c_block = tl.program_id(1)
    out_block = tl.program_id(2)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_active = out_offsets < (OUT_H * OUT_W)
    channel_active = channels < C
    out_h = out_offsets // OUT_W
    out_w = out_offsets - out_h * OUT_W

    mean = tl.load(mean_ptr + channels, mask=channel_active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=channel_active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_active, other=0.0).to(tl.float32)
    input_base = n_idx * stride_n + channels[:, None] * stride_c

    best = tl.full((BLOCK_C, BLOCK_OUT), -float("inf"), tl.float32)
    best_offset = tl.zeros((BLOCK_C, BLOCK_OUT), tl.int32)
    for kh in tl.static_range(0, 3):
        in_h = out_h * 2 + kh - 1
        valid_h = (in_h >= 0) & (in_h < H)
        load_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
        for kw in tl.static_range(0, 3):
            in_w = out_w * 2 + kw - 1
            valid_out = out_active & valid_h & (in_w >= 0) & (in_w < W)
            load_w = tl.minimum(tl.maximum(in_w, 0), W - 1)
            valid = channel_active[:, None] & valid_out[None, :]
            x = tl.load(
                x_ptr + input_base + load_h[None, :] * stride_h + load_w[None, :] * stride_w,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            centered = _f32_sub(x, mean[:, None])
            normalized = _f32_mul(centered, invstd[:, None])
            scaled = _f32_mul(normalized, weight[:, None])
            affine = _f32_add(scaled, bias[:, None])
            relu = _relu_preserve_nan(affine.to(tl.bfloat16).to(tl.float32))
            take = valid & ((relu > best) | ((relu != relu) & (best == best)))
            best = tl.where(take, relu, best)
            best_offset = tl.where(take, kh * 3 + kw, best_offset)

    out_hw = OUT_H * OUT_W
    out_base = n_idx * C * out_hw + channels[:, None] * out_hw + out_offsets[None, :]
    store_mask = channel_active[:, None] & out_active[None, :]
    tl.store(values_ptr + out_base, best, mask=store_mask)
    tl.store(offsets_ptr + out_base, best_offset.to(tl.int8), mask=store_mask)

    vals = tl.where(store_mask, best, 0.0)
    zero = tl.zeros([BLOCK_C, BLOCK_OUT], tl.float32)
    weights = tl.where(store_mask, 1.0, 0.0)
    block_mean, block_m2, block_weight = triton_helpers.welford(
        vals,
        zero,
        weights,
        1,
    )
    chunk = n_idx * NUM_OUT_BLOCKS + out_block
    partial_offsets = chunk * C + channels
    tl.store(partial_mean_ptr + partial_offsets, block_mean, mask=channel_active)
    tl.store(partial_m2_ptr + partial_offsets, block_m2, mask=channel_active)
    tl.store(partial_weight_ptr + partial_offsets, block_weight, mask=channel_active)

@triton.jit
def _bn_finalize_affine_relu_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    invstd_ptr,
    values_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    channel = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunks < NUM_CHUNKS
    offsets = chunks * C + channel
    block_mean = tl.load(partial_mean_ptr + offsets, mask=chunk_mask, other=0.0).to(tl.float32)
    block_m2 = tl.load(partial_m2_ptr + offsets, mask=chunk_mask, other=0.0).to(tl.float32)
    block_weight = tl.load(partial_weight_ptr + offsets, mask=chunk_mask, other=0.0).to(tl.float32)
    mean, m2, _weight = triton_helpers.welford(block_mean, block_m2, block_weight, 0)
    var = m2 / E
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    updated_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(var, 1.0000797257434426)
    updated_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))
    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)
    tl.store(running_mean_ptr + channel, updated_mean)
    tl.store(running_var_ptr + channel, updated_var)

    e_offsets = tl.arange(0, BLOCK_E)
    mask = e_offsets < E
    n_idx = e_offsets // HW
    hw = e_offsets - n_idx * HW
    dense_offsets = n_idx * C * HW + channel * HW + hw
    x = tl.load(values_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    y = _f32_add(scaled, bias).to(tl.bfloat16)
    y = _relu_preserve_nan(y.to(tl.float32)).to(tl.bfloat16)
    tl.store(y_ptr + dense_offsets, y, mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# (T([4,64,112,112], bf16), T([64], f32), T([64], f32), ...)
@oracle_impl(
    hardware="B200",
    point="45fdeec8",
    BLOCK_E1=2048,
    C_BLOCK_STATS=2,
    POOL_BLOCK_C=4,
    POOL_BLOCK_OUT=128,
    SECOND_BLOCK_E=16384,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_E1: int,
    C_BLOCK_STATS: int,
    POOL_BLOCK_C: int,
    POOL_BLOCK_OUT: int,
    SECOND_BLOCK_E: int,
):
    (
        x,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        running_mean2,
        running_var2,
        weight2,
        bias2,
        _kernel_size,
        _stride,
    ) = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    out_h = 56
    out_w = 56
    e1 = n * h * w
    e2 = n * out_h * out_w
    num_chunks1 = triton.cdiv(e1, BLOCK_E1)
    block_chunks1 = _next_power_of_2(num_chunks1)
    num_out_blocks = triton.cdiv(out_h * out_w, POOL_BLOCK_OUT)
    num_chunks2 = n * num_out_blocks
    block_chunks2 = _next_power_of_2(num_chunks2)

    partial_mean1 = torch.empty((num_chunks1, c), device=x.device, dtype=torch.float32)
    partial_m21 = torch.empty((num_chunks1, c), device=x.device, dtype=torch.float32)
    partial_weight1 = torch.empty((num_chunks1, c), device=x.device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd1 = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)

    values = torch.empty_strided(
        (n, c, out_h, out_w),
        (c * out_h * out_w, out_h * out_w, out_w, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    offsets = torch.empty_strided(
        (n, c, out_h, out_w),
        (c * out_h * out_w, out_h * out_w, out_w, 1),
        device=x.device,
        dtype=torch.int8,
    )
    partial_mean2 = torch.empty((num_chunks2, c), device=x.device, dtype=torch.float32)
    partial_m22 = torch.empty((num_chunks2, c), device=x.device, dtype=torch.float32)
    partial_weight2 = torch.empty((num_chunks2, c), device=x.device, dtype=torch.float32)
    mean2_vec = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    invstd2_4d = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    relu2 = torch.empty_strided(
        (n, c, out_h, out_w),
        (c * out_h * out_w, out_h * out_w, out_w, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _bn_partial_stats_kernel[(triton.cdiv(c, C_BLOCK_STATS), num_chunks1)](
        x,
        partial_mean1,
        partial_m21,
        partial_weight1,
        C=c,
        H=h,
        W=w,
        E=e1,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_E=BLOCK_E1,
        C_BLOCK=C_BLOCK_STATS,
        num_warps=8,
        num_stages=4,
    )
    _bn_finalize_stats_kernel[(triton.cdiv(c, C_BLOCK_STATS),)](
        partial_mean1,
        partial_m21,
        partial_weight1,
        running_mean1,
        running_var1,
        mean1,
        invstd1,
        C=c,
        E=e1,
        NUM_CHUNKS=num_chunks1,
        BLOCK_CHUNKS=block_chunks1,
        C_BLOCK=C_BLOCK_STATS,
        RUNNING_VAR_CORRECTION=1.0000199302441455,
        num_warps=8 if C_BLOCK_STATS * block_chunks1 >= 512 else 4,
        num_stages=4,
    )
    _bn_relu_maxpool3_partial_kernel[
        (n, triton.cdiv(c, POOL_BLOCK_C), num_out_blocks)
    ](
        x,
        weight1,
        bias1,
        mean1,
        invstd1,
        values,
        offsets,
        partial_mean2,
        partial_m22,
        partial_weight2,
        C=c,
        H=h,
        W=w,
        OUT_H=out_h,
        OUT_W=out_w,
        NUM_OUT_BLOCKS=num_out_blocks,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_C=POOL_BLOCK_C,
        BLOCK_OUT=POOL_BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )
    _bn_finalize_affine_relu_kernel[(c,)](
        partial_mean2,
        partial_m22,
        partial_weight2,
        running_mean2,
        running_var2,
        mean2_vec,
        invstd2_4d,
        values,
        weight2,
        bias2,
        relu2,
        C=c,
        HW=out_h * out_w,
        E=e2,
        NUM_CHUNKS=num_chunks2,
        BLOCK_CHUNKS=block_chunks2,
        BLOCK_E=SECOND_BLOCK_E,
        num_warps=8,
        num_stages=4,
    )

    return (
        mean1,
        invstd1,
        values,
        offsets,
        invstd2_4d.reshape((c,)),
        relu2,
        mean2_vec[None, :, None, None],
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )
