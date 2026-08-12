"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 training-BatchNorm ReLU spatial-mean scope in one channel-specialized Triton kernel, including bf16-to-fp32 population `var_mean(..., correction=0)` over N/H/W, eps=1e-5 rsqrt, returned saved-mean and rsqrt tensors, mutable running-stat `copy_` aliases with the captured variance correction, fp32 affine, explicit bf16 rounding before ReLU, and the final bf16 `[N,C]` spatial mean, whereas Inductor lowers the canonicalized var_mean/update/affine/cast/ReLU/mean graph through generic normalization and downstream reduction schedules; Inductor cannot do this today because its norm-template scheduler does not sink the immediate spatial-mean consumer and mutable running-stat epilogues into one small fixed-shape training-BN plan while preserving bf16 cast boundaries and side-output aliases; the fix is SCHEDULER_FUSION: extend the BN-training template to emit saved stats and running-stat side effects while fusing fixed-shape affine ReLU plus spatial-mean consumers into the channel-statistics lowering."""

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
def _bn_relu_spatial_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    spatial_mean_ptr,
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
    row_mean, row_m2, row_weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    mean, m2, _weight = triton_helpers.welford(row_mean, row_m2, row_weight, 0)
    var = m2 / E
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(old_mean, 0.9), _f32_mul(mean, 0.1))
    corrected_var = _f32_mul(var, RUNNING_VAR_CORRECTION)
    new_var = _f32_add(_f32_mul(old_var, 0.9), _f32_mul(corrected_var, 0.1))
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)
    tl.store(mean_ptr + channel, mean)
    tl.store(invstd_ptr + channel, invstd)

    gamma = tl.load(weight_ptr + channel).to(tl.float32)
    beta = tl.load(bias_ptr + channel).to(tl.float32)
    centered = _f32_sub(vals, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, gamma)
    biased = _f32_add(scaled, beta)
    rounded = biased.to(tl.bfloat16)
    relu = tl.where(rounded != rounded, rounded, tl.maximum(rounded, 0.0)).to(tl.bfloat16)
    pooled = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1) / hw_size
    tl.store(
        spatial_mean_ptr + n_offsets * C + channel,
        pooled.to(tl.bfloat16),
        mask=n_offsets < (E // hw_size),
    )


# 1a932376: (T([4,1024,7,7], bf16), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), S([4,1024]))
@oracle_impl(hardware="B200", point="1a932376", BLOCK_N=4, BLOCK_HW=64, num_warps=1)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_HW: int, num_warps: int):
    x, running_mean, running_var, weight, bias, _shape_param_0 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    e = n * h * w

    mean = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    spatial_mean = torch.empty_strided(
        (n, c),
        (c, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    running_var_correction = e / (e - 1.0)

    _bn_relu_spatial_mean_kernel[(c,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        mean,
        invstd,
        spatial_mean,
        C=c,
        H=h,
        W=w,
        E=e,
        STRIDE_N=x.stride(0),
        STRIDE_C=x.stride(1),
        STRIDE_H=x.stride(2),
        STRIDE_W=x.stride(3),
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        RUNNING_VAR_CORRECTION=running_var_correction,
        num_warps=num_warps,
        num_stages=3,
    )
    return mean, invstd, spatial_mean, running_mean, running_var
