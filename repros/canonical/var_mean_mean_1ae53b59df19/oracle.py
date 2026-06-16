"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full bf16 DenseNet cat plus training-BatchNorm ReLU spatial-mean scope, including the returned concatenated bf16 tensor, bf16-to-fp32 population statistics over N/H/W, saved mean and rsqrt side outputs, mutable running-stat copy_ aliases with the captured correction literal, fp32 affine math, bf16 rounding before ReLU, and final bf16 `[128,184]` spatial mean view, whereas Inductor lowers the cat, BN-training statistics/update, side outputs, ReLU, and downstream spatial mean through generic producer/consumer schedules; Inductor cannot do this today because its scheduler does not fuse a normalization template fed by a channel cat with a following reduction consumer while preserving mutable stat side effects, saved-stat returns, bf16 cast boundaries, and the returned cat; the fix is SCHEDULER_FUSION: add a BN-training producer schedule that can expose saved mean/invstd, honor running-stat aliases, and sink the affine/ReLU/spatial-mean consumer while materializing required cat output directly."""

import torch
import triton
import triton.language as tl
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
def _cat_bn_relu_spatial_mean_kernel(
    x0_ptr,
    x1_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    cat_out_ptr,
    mean_out_ptr,
    invstd_out_ptr,
    pooled_out_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    E: tl.constexpr,
    X0_SN: tl.constexpr,
    X0_SC: tl.constexpr,
    X0_SH: tl.constexpr,
    X0_SW: tl.constexpr,
    X1_SN: tl.constexpr,
    X1_SC: tl.constexpr,
    X1_SH: tl.constexpr,
    X1_SW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    channel = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    hw_offsets = tl.arange(0, BLOCK_HW)
    h_offsets = hw_offsets // 4
    w_offsets = hw_offsets - h_offsets * 4
    mask = (n_offsets[:, None] < (E // HW)) & (hw_offsets[None, :] < HW)

    from_x0 = channel < C0
    x0_offsets = (
        n_offsets[:, None] * X0_SN
        + channel * X0_SC
        + h_offsets[None, :] * X0_SH
        + w_offsets[None, :] * X0_SW
    )
    x1_channel = channel - C0
    x1_offsets = (
        n_offsets[:, None] * X1_SN
        + x1_channel * X1_SC
        + h_offsets[None, :] * X1_SH
        + w_offsets[None, :] * X1_SW
    )
    vals0 = tl.load(x0_ptr + x0_offsets, mask=mask & from_x0, other=0.0).to(tl.float32)
    vals1 = tl.load(x1_ptr + x1_offsets, mask=mask & (channel >= C0), other=0.0).to(tl.float32)
    vals = _f32_add(vals0, vals1)

    cat_offsets = (n_offsets[:, None] * C + channel) * HW + hw_offsets[None, :]
    tl.store(cat_out_ptr + cat_offsets, vals.to(tl.bfloat16), mask=mask)

    active = tl.where(mask, vals, 0.0)
    row_sum = tl.sum(active, axis=1)
    row_sum_sq = tl.sum(_f32_mul(active, active), axis=1)
    total = tl.sum(row_sum, axis=0)
    total_sq = tl.sum(row_sum_sq, axis=0)
    mean = total / E
    var = tl.maximum(_f32_sub(total_sq / E, _f32_mul(mean, mean)), 0.0)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    tl.store(running_mean_ptr + channel, _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9)))
    tl.store(
        running_var_ptr + channel,
        _f32_add(_f32_mul(_f32_mul(var, 1.0004885197850513), 0.1), _f32_mul(old_var, 0.9)),
    )
    tl.store(mean_out_ptr + channel, mean)
    tl.store(invstd_out_ptr + channel, invstd)

    gamma = tl.load(weight_ptr + channel).to(tl.float32)
    beta = tl.load(bias_ptr + channel).to(tl.float32)
    centered = _f32_sub(vals, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, gamma)
    affine = _f32_add(scaled, beta).to(tl.bfloat16)
    zero = tl.full((BLOCK_N, BLOCK_HW), 0.0, tl.bfloat16)
    relu = tl.where(affine < zero, zero, affine)
    pooled = tl.sum(tl.where(mask, relu.to(tl.float32), 0.0), axis=1) * 0.0625
    tl.store(pooled_out_ptr + n_offsets * C + channel, pooled.to(tl.bfloat16), mask=n_offsets < (E // HW))


# d10f4d93: DenseNet cat -> training BN -> bf16 ReLU spatial mean.
@oracle_impl(hardware="B200", point="d10f4d93", BLOCK_N=128, BLOCK_HW=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_HW: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0 = inputs
    n = arg0_1.shape[0]
    c0 = arg0_1.shape[1]
    c1 = arg1_1.shape[1]
    c = c0 + c1
    hw = arg0_1.shape[2] * arg0_1.shape[3]
    e = n * hw

    cat = torch.empty_strided((n, c, 4, 4), (c * hw, hw, 4, 1), device=arg0_1.device, dtype=torch.bfloat16)
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=arg0_1.device, dtype=torch.float32)
    pooled = torch.empty_strided((n, c), (c, 1), device=arg0_1.device, dtype=torch.bfloat16)

    _cat_bn_relu_spatial_mean_kernel[(c,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        cat,
        mean,
        invstd,
        pooled,
        c0,
        c1,
        c,
        hw,
        e,
        arg0_1.stride(0),
        arg0_1.stride(1),
        arg0_1.stride(2),
        arg0_1.stride(3),
        arg1_1.stride(0),
        arg1_1.stride(1),
        arg1_1.stride(2),
        arg1_1.stride(3),
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return cat, mean, invstd, pooled, arg2_1, arg3_1
