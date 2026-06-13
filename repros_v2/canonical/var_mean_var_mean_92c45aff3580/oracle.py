"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 dual training-BatchNorm channel-shuffle scope, including both bf16-to-fp32 per-channel population `var_mean(..., correction=0)` reductions, eps=1e-5 rsqrt side outputs, running mean/variance copy_ side effects with the captured correction literal, bf16 affine rounding before ReLU, direct channel-shuffle clone layout materialization, split-view returns sharing one backing allocation, and four returned running-stat aliases. Inductor lowers the sibling BN-training reductions, mutable running-stat updates, cat/view/permute/clone/view channel shuffle, and split return contract through generic normalization and layout schedules; it cannot do this today because the scheduler does not preserve the fixed semantic channel-shuffle consumer across two BN-training reductions with observable stats and mutable side effects. The fix is SCHEDULER_FUSION: teach the BN-training/layout scheduler to fuse fixed branch reductions with direct channel-shuffle stores and side-effect running-stat returns."""

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
def _bn_relu_shuffle_branch_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    mean_out_ptr,
    rsqrt_out_ptr,
    shuffled_ptr,
    BRANCH: tl.constexpr,
    C: tl.constexpr,
    OUT_C: tl.constexpr,
    HW: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    n_idx = offsets // HW
    hw_idx = offsets - n_idx * HW
    x_offsets = n_idx * C * HW + channel * HW + hw_idx

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=0) / K
    centered = _f32_sub(x, mean)
    variance = tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=0) / K
    variance = tl.maximum(variance, 0.0)
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

    tl.store(mean_out_ptr + channel, mean)
    tl.store(rsqrt_out_ptr + channel, invstd)

    old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
    old_var = tl.load(running_var_ptr + channel).to(tl.float32)
    new_mean = _f32_add(_f32_mul(mean, 0.1), _f32_mul(old_mean, 0.9))
    corrected_var = _f32_mul(variance, 1.0000398612827361)
    new_var = _f32_add(_f32_mul(corrected_var, 0.1), _f32_mul(old_var, 0.9))
    tl.store(running_mean_ptr + channel, new_mean)
    tl.store(running_var_ptr + channel, new_var)

    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias).to(tl.bfloat16)
    relu = tl.where(affine != affine, affine, tl.maximum(affine, 0.0))

    shuffled_channel = channel * 2 + BRANCH
    out_offsets = n_idx * OUT_C * HW + shuffled_channel * HW + hw_idx
    tl.store(shuffled_ptr + out_offsets, relu, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 55c2f021: (T([128,116,14,14], bf16), T([116], f32), T([116], f32), T([116], f32), T([116], f32), T([128,116,14,14], bf16), ...)
@oracle_impl(hardware="B200", point="55c2f021", BLOCK_K=32768, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int, num_stages: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        _shape_param_0,
        _shape_param_1,
    ) = inputs
    del _shape_param_0

    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    k = n * hw
    out_c = 2 * c
    mean_shape = (1, c, 1, 1)
    mean_stride = _contiguous_stride(mean_shape)

    mean0 = torch.empty_strided(mean_shape, mean_stride, device=arg0_1.device, dtype=torch.float32)
    rsqrt0 = torch.empty_strided(mean_shape, mean_stride, device=arg0_1.device, dtype=torch.float32)
    mean1 = torch.empty_strided(mean_shape, mean_stride, device=arg0_1.device, dtype=torch.float32)
    rsqrt1 = torch.empty_strided(mean_shape, mean_stride, device=arg0_1.device, dtype=torch.float32)
    shuffled = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        (out_c * hw, hw, w, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _bn_relu_shuffle_branch_kernel[(c,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        mean0,
        rsqrt0,
        shuffled,
        BRANCH=0,
        C=c,
        OUT_C=out_c,
        HW=hw,
        K=k,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _bn_relu_shuffle_branch_kernel[(c,)](
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        mean1,
        rsqrt1,
        shuffled,
        BRANCH=1,
        C=c,
        OUT_C=out_c,
        HW=hw,
        K=k,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    first, second = torch.split(shuffled, c, 1)
    return mean0, rsqrt0, mean1, rsqrt1, first, second, arg1_1, arg2_1, arg6_1, arg7_1
