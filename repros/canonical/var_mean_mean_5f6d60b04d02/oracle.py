"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 ResNet training-BatchNorm residual-ReLU plus spatial-mean scope in one channel-tiled Triton program, including the bf16-to-fp32 input promotion, population `var_mean(..., correction=0)`, eps=1e-5 `rsqrt`, saved invstd and mean side outputs, mutable running-stat `copy_` outputs, affine bf16 rounding before the residual add, ReLU, final `[N,C]` bf16 spatial mean view, and bool `le` mask, whereas Inductor lowers the canonicalized BN-training template and the downstream ReLU mean/mask consumers as separate scheduled regions with exposed intermediate activation traffic; Inductor cannot do this today because its normalization scheduler cannot sink multiple post-BN consumers and mutable stat epilogues into a single full-scope channel reduction while preserving bf16 cast boundaries and side-output aliases; the fix is SCHEDULER_FUSION: extend the BN-training scheduler template to emit mean/invstd/running-stat side outputs and fuse residual affine, ReLU, spatial mean, and mask consumers into the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


RUNNING_VAR_CORRECTION = 1.0012771392081736


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
def _bn_relu_spatial_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    spatial_mean_ptr,
    le_mask_ptr,
    mean_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_C: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
):
    channel = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    hw_offsets = tl.arange(0, BLOCK_HW)
    hw_size: tl.constexpr = H * W
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets - h_offsets * W
    mask = (n_offsets[:, None] < (E // hw_size)) & (hw_offsets[None, :] < hw_size)
    input_offsets = (
        n_offsets[:, None] * STRIDE_N
        + channel * STRIDE_C
        + h_offsets[None, :] * STRIDE_H
        + w_offsets[None, :] * STRIDE_W
    )

    vals = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([BLOCK_N, BLOCK_HW], tl.float32)
    m2_acc = tl.zeros([BLOCK_N, BLOCK_HW], tl.float32)
    weight_acc = tl.zeros([BLOCK_N, BLOCK_HW], tl.float32)
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
    row_mean, row_m2, row_weight = triton_helpers.welford(
        mean_acc,
        m2_acc,
        weight_acc,
        1,
    )
    mean, m2, _weight = triton_helpers.welford(row_mean, row_m2, row_weight, 0)
    var = m2 / E
    invstd = libdevice.rsqrt(var + 1.0e-5)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(running_mean_ptr + channel, old_mean * 0.9 + mean * 0.1)
    tl.store(
        running_var_ptr + channel,
        old_var * 0.9 + var * RUNNING_VAR_CORRECTION * 0.1,
    )
    tl.store(invstd_ptr + channel, invstd)
    tl.store(mean_ptr + channel, mean)

    gamma = tl.load(weight_ptr + channel).to(tl.float32)
    beta = tl.load(bias_ptr + channel).to(tl.float32)
    residual = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0)
    centered = _f32_sub(vals, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, gamma)
    biased = _f32_add(scaled, beta)
    y = biased.to(tl.bfloat16)
    y = (y + residual).to(tl.bfloat16)
    zero = tl.full([BLOCK_N, BLOCK_HW], 0.0, tl.bfloat16)
    relu = tl.where(y < zero, zero, y)

    contiguous_offsets = (n_offsets[:, None] * C + channel) * hw_size + hw_offsets[None, :]
    tl.store(le_mask_ptr + contiguous_offsets, relu <= zero, mask=mask)

    pooled = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1) / hw_size
    tl.store(
        spatial_mean_ptr + n_offsets * C + channel,
        pooled.to(tl.bfloat16),
        mask=n_offsets < (E // hw_size),
    )


@triton.jit
def _bn_stats_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    invstd_ptr,
    mean_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_C: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    BLOCK_E: tl.constexpr,
    RUNNING_VAR_CORRECTION: tl.constexpr,
):
    channel = tl.program_id(0)
    e_offsets = tl.arange(0, BLOCK_E)[None, :]
    hw_size: tl.constexpr = H * W
    n_offsets = e_offsets // hw_size
    hw_offsets = e_offsets - n_offsets * hw_size
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets - h_offsets * W
    mask = e_offsets < E
    input_offsets = (
        n_offsets * STRIDE_N
        + channel * STRIDE_C
        + h_offsets * STRIDE_H
        + w_offsets * STRIDE_W
    )

    vals = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([1, BLOCK_E], tl.float32)
    m2_acc = tl.zeros([1, BLOCK_E], tl.float32)
    weight_acc = tl.zeros([1, BLOCK_E], tl.float32)
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
    scalar = tl.arange(0, 1)

    old_mean = tl.load(running_mean_ptr + channel + scalar).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel + scalar).to(tl.float32)
    tl.store(running_mean_ptr + channel + scalar, old_mean * 0.9 + mean * 0.1)
    tl.store(
        running_var_ptr + channel + scalar,
        old_var * 0.9 + var * RUNNING_VAR_CORRECTION * 0.1,
    )
    tl.store(invstd_ptr + channel + scalar, invstd)
    tl.store(mean_ptr + channel + scalar, mean)


@triton.jit
def _bn_epilogue_spatial_mean_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    invstd_ptr,
    mean_ptr,
    spatial_mean_ptr,
    le_mask_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_C: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw_offsets = tl.arange(0, BLOCK_HW)
    hw_size: tl.constexpr = H * W
    n_offsets = rows // C
    channels = rows - n_offsets * C
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets - h_offsets * W
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < hw_size
    mask = row_mask[:, None] & hw_mask[None, :]
    input_offsets = (
        n_offsets[:, None] * STRIDE_N
        + channels[:, None] * STRIDE_C
        + h_offsets[None, :] * STRIDE_H
        + w_offsets[None, :] * STRIDE_W
    )

    x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    gamma = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    beta = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0)

    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, gamma[:, None])
    biased = _f32_add(scaled, beta[:, None])
    y = biased.to(tl.bfloat16)
    y = (y + residual).to(tl.bfloat16)
    zero = tl.full([BLOCK_ROWS, BLOCK_HW], 0.0, tl.bfloat16)
    relu = tl.where(y < zero, zero, y)

    contiguous_offsets = rows[:, None] * hw_size + hw_offsets[None, :]
    tl.store(le_mask_ptr + contiguous_offsets, relu <= zero, mask=mask)
    pooled = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1) / hw_size
    tl.store(spatial_mean_ptr + rows, pooled.to(tl.bfloat16), mask=row_mask)


# 79146166: (T([16,512,7,7], bf16), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([16,512,7,7], bf16), S([16,512]))
@oracle_impl(hardware="B200", point="79146166")
# 8881253b: (T([32,2048,7,7], bf16), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([32,2048,7,7], bf16), S([32,2048]))
@oracle_impl(hardware="B200", point="8881253b")
# c99f0cec: (T([8,2048,7,7], bf16), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([8,2048,7,7], bf16), S([8,2048]))
@oracle_impl(hardware="B200", point="c99f0cec")
# 77734290: (T([128,64,8,8], bf16), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([128,64,8,8], bf16), S([128,64]))
@oracle_impl(hardware="B200", point="77734290")
# f7eda15e: (T([128,2048,7,7], bf16), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([128,2048,7,7], bf16), S([128,2048]))
@oracle_impl(hardware="B200", point="f7eda15e")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, residual, _shape_param_0 = inputs
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    e = n * hw
    block_n = triton.next_power_of_2(n)
    block_hw = triton.next_power_of_2(hw)

    invstd = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    spatial_mean = torch.empty_strided((n, c), (c, 1), device=x.device, dtype=torch.bfloat16)
    le_mask = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bool)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32)

    if n == 32 and c == 2048 and h == 7 and w == 7:
        _bn_stats_kernel[(c,)](
            x,
            running_mean,
            running_var,
            invstd,
            mean,
            h,
            w,
            e,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_E=2048,
            RUNNING_VAR_CORRECTION=RUNNING_VAR_CORRECTION,
            num_warps=4,
            num_stages=3,
        )
        _bn_epilogue_spatial_mean_kernel[(triton.cdiv(n * c, 16),)](
            x,
            weight,
            bias,
            residual,
            invstd,
            mean,
            spatial_mean,
            le_mask,
            c,
            h,
            w,
            n * c,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_ROWS=16,
            BLOCK_HW=64,
            num_warps=1,
            num_stages=3,
        )
        return invstd, spatial_mean, le_mask, mean, running_mean, running_var

    _bn_relu_spatial_mean_kernel[(c,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        residual,
        invstd,
        spatial_mean,
        le_mask,
        mean,
        c,
        h,
        w,
        e,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_N=block_n,
        BLOCK_HW=block_hw,
        RUNNING_VAR_CORRECTION=RUNNING_VAR_CORRECTION,
        num_warps=1 if block_n * block_hw <= 2048 else 8,
        num_stages=3,
    )
    return invstd, spatial_mean, le_mask, mean, running_mean, running_var
