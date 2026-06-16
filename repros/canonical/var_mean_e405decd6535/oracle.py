"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 GroupNorm-style inference block in one shape-specialized Triton row kernel, including bf16-to-fp32 activation promotion, correction=0 `var_mean` over each 32-group row, eps=1e-5 rsqrt normalization, bf16 affine scale/bias promotion, the required affine-to-bf16 rounding before residual add, bf16 residual add rounding, and final bf16 ReLU output, whereas Inductor lowers the decomposed view/var_mean/affine/cast/residual/ReLU graph through its generic normalization scheduler; Inductor cannot do this today because its pattern library does not select a guarded small-group GroupNorm-affine-residual-ReLU template that strips generic reduction scaffolding while preserving the bf16 cast boundaries; the fix is NEW_PATTERN: add a small-group GroupNorm inference lowering that fuses the affine and residual epilogue into the normalization row plan with explicit dtype-boundary guards."""

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
def _groupnorm_affine_residual_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    TOTAL_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    GROUP_ELEMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    elems = tl.arange(0, BLOCK_K)
    mask = (rows[:, None] < TOTAL_GROUPS) & (elems[None, :] < GROUP_ELEMS)
    offsets = rows[:, None] * GROUP_ELEMS + elems[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    active_cols = elems[None, :] < GROUP_ELEMS
    mean = tl.sum(tl.where(active_cols, x, 0.0), axis=1) / GROUP_ELEMS
    centered = _f32_sub(x, mean[:, None])
    centered_for_reduce = tl.where(active_cols, centered, 0.0)
    variance = (
        tl.sum(_f32_mul(centered_for_reduce, centered_for_reduce), axis=1)
        / GROUP_ELEMS
    )
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))
    normalized = _f32_mul(centered, invstd[:, None])

    channels_per_group: tl.constexpr = CHANNELS // 32
    group_id = rows % 32
    channel = group_id[:, None] * channels_per_group + elems[None, :] // HW
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rounded_affine = affine.to(tl.bfloat16).to(tl.float32)
    rounded_sum = _f32_add(rounded_affine, residual).to(tl.bfloat16).to(tl.float32)
    relu = tl.where((rounded_sum > 0.0) | (rounded_sum != rounded_sum), rounded_sum, 0.0)
    tl.store(out_ptr + offsets, relu.to(tl.bfloat16), mask=mask)


# (T([128,256,2,2], bf16), T([256], bf16), T([256], bf16), T([128,256,2,2], bf16), S([128,32,8,4]), S([128,256,2,2]))
@oracle_impl(hardware="B200", point="49e390ef", BLOCK_M=4, BLOCK_K=32, num_warps=1, num_stages=3)
# (T([128,128,4,4], bf16), T([128], bf16), T([128], bf16), T([128,128,4,4], bf16), S([128,32,4,16]), S([128,128,4,4]))
@oracle_impl(hardware="B200", point="d6ffe01a", BLOCK_M=4, BLOCK_K=64, num_warps=1, num_stages=3)
# (T([128,64,8,8], bf16), T([64], bf16), T([64], bf16), T([128,64,8,8], bf16), S([128,32,2,64]), S([128,64,8,8]))
@oracle_impl(hardware="B200", point="f9823638", BLOCK_M=2, BLOCK_K=128, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_K, num_warps, num_stages):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    group_elems = (channels // 32) * hw
    total_groups = batch * 32

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        tuple(int(stride) for stride in arg3_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _groupnorm_affine_residual_relu_kernel[(triton.cdiv(total_groups, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        TOTAL_GROUPS=total_groups,
        CHANNELS=channels,
        HW=hw,
        GROUP_ELEMS=group_elems,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
