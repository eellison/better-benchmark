"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 ResNet/Shufflenet training-BatchNorm var_mean scope, including the bf16-to-fp32 population statistics, returned mean and rsqrt side outputs, mutable running-stat copy_ aliases with the captured correction literal, exact bf16 affine/ReLU cast boundary, and low-memory 3x3 stride-2 maxpool values and int8 offsets from one fused normalization-to-pool stencil, whereas Inductor lowers the statistics, running-stat epilogues, bf16 activation materialization, and maxpool-with-offsets consumer as separate generic schedules; Inductor cannot do this today because scheduler fusion does not inline a reduction producer with mutation side outputs into a multi-output low-memory maxpool consumer while preserving bf16 rounding and offset semantics; the fix is SCHEDULER_FUSION: teach the training-BatchNorm scheduler to emit running-stat side effects and sink the exact bf16 affine/ReLU producer into the fixed maxpool-with-offsets stencil."""

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
    corrected_var = _f32_mul(var, 1.0000049824865598)
    updated_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))

    tl.store(mean_ptr + channels, mean, mask=channel_mask)
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(running_mean_ptr + channels, updated_mean, mask=channel_mask)
    tl.store(running_var_ptr + channels, updated_var, mask=channel_mask)


@triton.jit
def _bn_relu_maxpool3_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    values_ptr,
    offsets_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
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


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# 5e4f83d3: (T([16,64,112,112], bf16), T([64], f32), T([64], f32), T([64], f32), T([64], f32), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="5e4f83d3", BLOCK_E=2048, C_BLOCK=1, POOL_BLOCK_C=2, POOL_BLOCK_OUT=128)
# 22d6d5c7: (T([32,64,112,112], bf16), T([64], f32), T([64], f32), T([64], f32), T([64], f32), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="22d6d5c7", BLOCK_E=2048, C_BLOCK=1, POOL_BLOCK_C=2, POOL_BLOCK_OUT=128)
# 4ee39e3f: (T([8,64,112,112], bf16), T([64], f32), T([64], f32), T([64], f32), T([64], f32), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="4ee39e3f", BLOCK_E=2048, C_BLOCK=1, POOL_BLOCK_C=2, POOL_BLOCK_OUT=128)
# 75543dfe: (T([128,24,112,112], bf16), T([24], f32), T([24], f32), T([24], f32), T([24], f32), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="75543dfe", BLOCK_E=2048, C_BLOCK=1, POOL_BLOCK_C=2, POOL_BLOCK_OUT=128)
# eaa8fe86: (T([128,64,112,112], bf16), T([64], f32), T([64], f32), T([64], f32), T([64], f32), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="eaa8fe86", BLOCK_E=4096, C_BLOCK=1, POOL_BLOCK_C=2, POOL_BLOCK_OUT=128)
def oracle_forward(
    inputs,
    *,
    BLOCK_E: int,
    C_BLOCK: int,
    POOL_BLOCK_C: int,
    POOL_BLOCK_OUT: int,
):
    x, running_mean, running_var, weight, bias, _kernel_size, _stride = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    out_h = 56
    out_w = 56
    e = n * h * w
    num_chunks = triton.cdiv(e, BLOCK_E)
    block_chunks = _next_power_of_2(num_chunks)

    partial_mean = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_m2 = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_weight = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)
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

    _bn_partial_stats_kernel[(triton.cdiv(c, C_BLOCK), num_chunks)](
        x,
        partial_mean,
        partial_m2,
        partial_weight,
        C=c,
        H=h,
        W=w,
        E=e,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_E=BLOCK_E,
        C_BLOCK=C_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _bn_finalize_stats_kernel[(triton.cdiv(c, C_BLOCK),)](
        partial_mean,
        partial_m2,
        partial_weight,
        running_mean,
        running_var,
        mean,
        invstd,
        C=c,
        E=e,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=C_BLOCK,
        num_warps=8 if C_BLOCK * block_chunks >= 512 else 4,
        num_stages=4,
    )
    _bn_relu_maxpool3_kernel[(n, triton.cdiv(c, POOL_BLOCK_C), triton.cdiv(out_h * out_w, POOL_BLOCK_OUT))](
        x,
        weight,
        bias,
        mean,
        invstd,
        values,
        offsets,
        C=c,
        H=h,
        W=w,
        OUT_H=out_h,
        OUT_W=out_w,
        stride_n=int(x.stride(0)),
        stride_c=int(x.stride(1)),
        stride_h=int(x.stride(2)),
        stride_w=int(x.stride(3)),
        BLOCK_C=POOL_BLOCK_C,
        BLOCK_OUT=POOL_BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )
    return mean, invstd, values, offsets, running_mean, running_var
