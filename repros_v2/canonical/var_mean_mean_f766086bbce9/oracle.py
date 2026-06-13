"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 singleton-spatial GroupNorm residual mean scope in one shape-specialized Triton row kernel, including bf16-to-fp32 activation promotion, fixed-32-group population var_mean with eps=1e-5 rsqrt, bf16 affine scale/bias promotion, the required affine-to-bf16 rounding before residual add, bf16 residual-add rounding, bf16 ReLU, and the final singleton spatial mean/view as a direct contiguous [N,C] store, whereas Inductor lowers the decomposed view/var_mean/sub/rsqrt/affine/cast/residual/ReLU/mean/view graph through generic normalization and reduction scheduling; Inductor cannot do this today because its pattern library does not recognize this fixed-group GroupNorm residual block with a size-one spatial mean consumer as one full-scope template with explicit bf16 rounding boundaries; the fix is NEW_PATTERN: add a guarded small-group GroupNorm residual lowering that keeps row statistics in registers and sinks singleton mean/view emission into the epilogue."""

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
def _groupnorm_residual_mean_kernel(
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
    active_cols = elems[None, :] < GROUP_ELEMS
    offsets = rows[:, None] * GROUP_ELEMS + elems[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(active_cols, x, 0.0), axis=1) / GROUP_ELEMS
    centered = _f32_sub(x, mean[:, None])
    centered_for_reduce = tl.where(active_cols, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_reduce, centered_for_reduce), axis=1) / GROUP_ELEMS
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


# cca94b79: (T([128,512,1,1], bf16), T([512], bf16), T([512], bf16), T([128,512,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="cca94b79", BLOCK_M=16, BLOCK_K=16, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    group_elems = channels // 32 * hw
    total_groups = batch * 32

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        (channels, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _groupnorm_residual_mean_kernel[(triton.cdiv(total_groups, BLOCK_M),)](
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
