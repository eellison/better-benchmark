"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 training-BatchNorm residual-ReLU scope, including the input bf16-to-fp32 cast, population var_mean over `[N,H,W]`, eps=1e-5 rsqrt, `[C]` invstd output, `[1,C,1,1]` saved-mean output, in-place running mean/variance copy_ updates with the captured correction constant, bf16 affine rounding before the residual add, and the full bf16 ReLU activation store, by splitting each per-channel statistics reduction into cooperative chunks before a compact finalizer and dense residual epilogue, whereas Inductor lowers the canonicalized norm-template graph as generic statistics and pointwise schedules; Inductor cannot do this today because the normalization scheduler lacks a guarded cooperative split-K training-BatchNorm variant that also preserves mutable running-stat aliases, side outputs, bf16 cast boundaries, and the residual-ReLU consumer; the fix is COOPERATIVE_SPLIT_K: add a BN-training stats template that splits the reduction dimension, finalizes invstd/mean/running-stat side outputs, and feeds the residual affine bf16 epilogue from the same schedule."""

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
    hw = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    m2_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    weight_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        vals,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    block_mean, block_m2, block_weight = triton_helpers.welford(
        mean_acc,
        m2_acc,
        weight_acc,
        1,
    )
    out_offsets = e_block * C + channels
    tl.store(partial_mean_ptr + out_offsets, block_mean, mask=channels < C)
    tl.store(partial_m2_ptr + out_offsets, block_m2, mask=channels < C)
    tl.store(partial_weight_ptr + out_offsets, block_weight, mask=channels < C)


@triton.jit
def _bn_finalize_stats_kernel(
    partial_mean_ptr,
    partial_m2_ptr,
    partial_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_ptr,
    mean_ptr,
    C: tl.constexpr,
    E: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
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
    mean_inputs = tl.where(mask, block_mean, 0.0)
    m2_inputs = tl.where(mask, block_m2, 0.0)
    weight_inputs = tl.where(mask, block_weight, 0.0)
    mean, m2, _weight = triton_helpers.welford(mean_inputs, m2_inputs, weight_inputs, 1)
    var = m2 / E
    invstd = libdevice.rsqrt(var + 1.0e-5)
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        old_var * 0.9 + var * RUNNING_VAR_CORRECTION * 0.1,
        mask=channel_mask,
    )
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(mean_ptr + channels, mean, mask=channel_mask)


@triton.jit
def _bn_affine_residual_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    mean_ptr,
    y_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    if CHANNELS_LAST:
        channels = offsets % C
    else:
        channels = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0)
    invstd = tl.load(invstd_ptr + channels, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    biased = _f32_add(scaled, bias)
    y = biased.to(tl.bfloat16)
    y = (y + residual).to(tl.bfloat16)
    y = tl.where(y != y, y, tl.maximum(y, 0.0))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _bn_single_pass_residual_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    y_ptr,
    mean_ptr,
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
    RUNNING_VAR_CORRECTION: tl.constexpr,
):
    c_block = tl.program_id(0)
    channels = c_block * C_BLOCK + tl.arange(0, C_BLOCK)
    e_offsets = tl.arange(0, BLOCK_E)
    hw = e_offsets % (H * W)
    n_idx = e_offsets // (H * W)
    h_idx = hw // W
    w_idx = hw - h_idx * W
    offsets = (
        n_idx[None, :] * stride_n
        + channels[:, None] * stride_c
        + h_idx[None, :] * stride_h
        + w_idx[None, :] * stride_w
    )
    mask = (channels[:, None] < C) & (e_offsets[None, :] < E)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    m2_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    weight_acc = tl.zeros([C_BLOCK, BLOCK_E], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        vals,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    var = m2 / E
    invstd = libdevice.rsqrt(var + 1.0e-5)
    channel_mask = channels < C

    old_mean = tl.load(running_mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=channel_mask)
    tl.store(
        running_var_ptr + channels,
        old_var * 0.9 + var * RUNNING_VAR_CORRECTION * 0.1,
        mask=channel_mask,
    )
    tl.store(invstd_ptr + channels, invstd, mask=channel_mask)
    tl.store(mean_ptr + channels, mean, mask=channel_mask)

    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    centered = _f32_sub(vals, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, weight[:, None])
    biased = _f32_add(scaled, bias[:, None])
    y = biased.to(tl.bfloat16)
    y = (y + residual).to(tl.bfloat16)
    y = tl.where(y != y, y, tl.maximum(y, 0.0))
    tl.store(y_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="2a5a9625")
@oracle_impl(hardware="B200", point="c772ba0f")
@oracle_impl(hardware="B200", point="8d0ce3d5")
@oracle_impl(hardware="B200", point="b2063b0a")
@oracle_impl(hardware="B200", point="f584d38d")
@oracle_impl(hardware="B200", point="912cb7ce")
@oracle_impl(hardware="B200", point="8881253b")
@oracle_impl(hardware="B200", point="4ace980f")
@oracle_impl(hardware="B200", point="0815ad0d")
@oracle_impl(hardware="B200", point="faf0eb9e")
@oracle_impl(hardware="B200", point="c99f0cec")
@oracle_impl(hardware="B200", point="b1e0eed3")
@oracle_impl(hardware="B200", point="56f0f473")
@oracle_impl(hardware="B200", point="39fb619f")
@oracle_impl(hardware="B200", point="77734290")
@oracle_impl(hardware="B200", point="a4d97c4c")
@oracle_impl(hardware="B200", point="79b5368f")
@oracle_impl(hardware="B200", point="f7eda15e")
@oracle_impl(hardware="B200", point="e7ff3dbe")
@oracle_impl(hardware="B200", point="60428e86")
@oracle_impl(hardware="B200", point="cd42bd92")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, residual = inputs
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    e = n * hw
    total = n * c * hw
    running_var_correction = 1.0001627869119323

    block_e = 1024
    if e <= 256:
        block_e = triton.next_power_of_2(e)
    elif e <= 512:
        block_e = 512
    c_block = 16 if e <= 1024 else 8
    num_chunks = triton.cdiv(e, block_e)
    block_chunks = triton.next_power_of_2(num_chunks)

    invstd = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    y = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)

    if e <= 8192:
        block_e = triton.next_power_of_2(e)
        c_block = 1
        _bn_single_pass_residual_relu_kernel[(triton.cdiv(c, c_block),)](
            x,
            running_mean,
            running_var,
            weight,
            bias,
            residual,
            invstd,
            y,
            mean,
            c,
            h,
            w,
            e,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_E=block_e,
            C_BLOCK=c_block,
            RUNNING_VAR_CORRECTION=running_var_correction,
            num_warps=1 if e <= 256 else (8 if c_block * block_e >= 8192 else 4),
        )
        return invstd, y, mean, running_mean, running_var

    partial_mean = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_m2 = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)
    partial_weight = torch.empty((num_chunks, c), device=x.device, dtype=torch.float32)

    _bn_partial_stats_kernel[(triton.cdiv(c, c_block), num_chunks)](
        x,
        partial_mean,
        partial_m2,
        partial_weight,
        c,
        h,
        w,
        e,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_E=block_e,
        C_BLOCK=c_block,
        num_warps=8 if c_block * block_e >= 8192 else 4,
    )
    _bn_finalize_stats_kernel[(triton.cdiv(c, c_block),)](
        partial_mean,
        partial_m2,
        partial_weight,
        running_mean,
        running_var,
        invstd,
        mean,
        c,
        e,
        num_chunks,
        BLOCK_E=block_e,
        BLOCK_CHUNKS=block_chunks,
        C_BLOCK=c_block,
        RUNNING_VAR_CORRECTION=running_var_correction,
        num_warps=8 if c_block * block_chunks >= 8192 else 4,
    )
    _bn_affine_residual_relu_kernel[(triton.cdiv(total, 1024),)](
        x,
        weight,
        bias,
        residual,
        invstd,
        mean,
        y,
        c,
        hw,
        total,
        CHANNELS_LAST=x.stride(1) == 1,
        BLOCK=1024,
        num_warps=4,
    )
    return invstd, y, mean, running_mean, running_var
