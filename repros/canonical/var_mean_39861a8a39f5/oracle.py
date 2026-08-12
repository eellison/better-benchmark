"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 GroupNorm-style residual ReLU training fragment in one shape-specialized Triton group-tile kernel, including bf16-to-fp32 input promotion, the metadata view into 32 groups, population `var_mean(..., dim=(2,3), correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned squeezed mean and rsqrt tensors, fp32 affine plus residual add, final ReLU, the required bf16 cast, and the bool `le(relu, 0)` mask, whereas Inductor lowers the decomposed graph through generic normalization, pointwise, cast, and mask schedules; Inductor cannot fuse this returned-output envelope today because its norm pattern library does not recognize this small fixed-32-group GroupNorm affine/residual/ReLU/mask idiom with exposed stats as one full-scope template while preserving f32 and bf16 output boundaries; the fix is NEW_PATTERN: add a guarded GroupNorm forward lowering that returns auxiliary stats and fuses affine residual ReLU, final bf16 cast, and ReLU-derived mask stores from one reduction tile."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


GROUPS = 32
EPS = 1.0e-5


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
def _groupnorm_residual_mask_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    mean_ptr,
    rsqrt_ptr,
    bf16_ptr,
    mask_ptr,
    TOTAL_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    GROUP_ELEMS: tl.constexpr,
    GROUPS_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    elems = tl.arange(0, BLOCK_K)
    row_mask = rows < TOTAL_GROUPS
    elem_mask = elems < GROUP_ELEMS
    mask = row_mask[:, None] & elem_mask[None, :]
    offsets = rows[:, None] * GROUP_ELEMS + elems[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=1) / GROUP_ELEMS
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / GROUP_ELEMS
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

    channels_per_group: tl.constexpr = CHANNELS // GROUPS_C
    group = rows % GROUPS_C
    channel = group[:, None] * channels_per_group + elems[None, :] // HW
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    affine = _f32_add(_f32_mul(normalized, weight), bias)
    summed = _f32_add(affine, residual)
    relu = tl.where((summed > 0.0) | (summed != summed), summed, 0.0)

    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(rsqrt_ptr + rows, invstd, mask=row_mask)
    tl.store(bf16_ptr + offsets, relu.to(tl.bfloat16), mask=mask)
    tl.store(mask_ptr + offsets, relu <= 0.0, mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _launch(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    group_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    groups = int(group_shape[1])
    group_elems = int(group_shape[2]) * int(group_shape[3])
    total_groups = batch * groups

    mean = torch.empty_strided(
        (batch, groups),
        (groups, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (batch, groups),
        (groups, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    bf16 = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    le = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    _groupnorm_residual_mask_kernel[(triton.cdiv(total_groups, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        mean,
        rsqrt,
        bf16,
        le,
        TOTAL_GROUPS=total_groups,
        CHANNELS=channels,
        HW=hw,
        GROUP_ELEMS=group_elems,
        GROUPS_C=groups,
        EPS_C=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mean, rsqrt, bf16, le


@oracle_impl(hardware="B200", point="af9e4718", BLOCK_M=8, BLOCK_K=32, num_warps=1, num_stages=3)
def oracle_forward_af9e4718(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int, num_stages: int):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages)


@oracle_impl(hardware="B200", point="b79ebf0a", BLOCK_M=8, BLOCK_K=64, num_warps=1, num_stages=3)
def oracle_forward_b79ebf0a(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int, num_stages: int):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages)


@oracle_impl(hardware="B200", point="a488e261", BLOCK_M=4, BLOCK_K=128, num_warps=4, num_stages=3)
def oracle_forward_a488e261(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int, num_stages: int):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages)


def oracle_forward(inputs):
    channels = int(inputs[0].shape[1])
    if channels == 256:
        return _launch(inputs, BLOCK_M=8, BLOCK_K=32, num_warps=1, num_stages=3)
    if channels == 128:
        return _launch(inputs, BLOCK_M=8, BLOCK_K=64, num_warps=1, num_stages=3)
    return _launch(inputs, BLOCK_M=4, BLOCK_K=128, num_warps=4, num_stages=3)
