"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 dual GroupNorm affine ReLU scope in one Triton row kernel, including both bf16-to-fp32 activation promotions, both correction=0 `var_mean(..., dim=(2,3), keepdim=True)` reductions over fixed 32-group views, eps-before-rsqrt normalization, bf16 affine scale/bias promotion, explicit bf16 rounding after each branch affine, final bf16 branch add, and bf16 ReLU output, whereas Inductor lowers the two norm-template reductions and the downstream cast/add/ReLU epilogue through generic scheduled normalization fragments; Inductor cannot fuse this full returned-output envelope today because its normalization scheduler does not co-schedule sibling small-group GroupNorm reductions while preserving per-branch bf16 cast boundaries and the final bf16 add/ReLU consumer; the fix is SCHEDULER_FUSION: extend the norm-template scheduler to group sibling fixed-shape GroupNorm reductions and emit the branch affine casts plus final add/ReLU output from one full-scope schedule."""

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
def _dual_groupnorm_affine_relu_kernel(
    x0_ptr,
    x1_ptr,
    weight0_ptr,
    bias0_ptr,
    weight1_ptr,
    bias1_ptr,
    out_ptr,
    TOTAL_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW_SIZE: tl.constexpr,
    GROUP_ELEMS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    elems = tl.arange(0, BLOCK_K)
    mask = (rows[:, None] < TOTAL_GROUPS) & (elems[None, :] < GROUP_ELEMS)
    offsets = rows[:, None] * GROUP_ELEMS + elems[None, :]

    x0 = tl.load(
        x0_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x1 = tl.load(
        x1_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    count = GROUP_ELEMS + 0.0
    mean0 = tl.sum(tl.where(mask, x0, 0.0), axis=1) / count
    mean1 = tl.sum(tl.where(mask, x1, 0.0), axis=1) / count
    centered0 = _f32_sub(x0, mean0[:, None])
    centered1 = _f32_sub(x1, mean1[:, None])
    centered0_masked = tl.where(mask, centered0, 0.0)
    centered1_masked = tl.where(mask, centered1, 0.0)
    var0 = tl.sum(_f32_mul(centered0_masked, centered0_masked), axis=1) / count
    var1 = tl.sum(_f32_mul(centered1_masked, centered1_masked), axis=1) / count
    inv0 = libdevice.rsqrt(_f32_add(var0, EPS))
    inv1 = libdevice.rsqrt(_f32_add(var1, EPS))
    norm0 = _f32_mul(centered0, inv0[:, None])
    norm1 = _f32_mul(centered1, inv1[:, None])

    group_id = rows % NUM_GROUPS
    channels_per_group: tl.constexpr = CHANNELS // NUM_GROUPS
    channel = group_id[:, None] * channels_per_group + elems[None, :] // HW_SIZE
    w0 = tl.load(
        weight0_ptr + channel,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    b0 = tl.load(
        bias0_ptr + channel,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    w1 = tl.load(
        weight1_ptr + channel,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    b1 = tl.load(
        bias1_ptr + channel,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    branch0 = _f32_add(_f32_mul(norm0, w0), b0).to(tl.bfloat16)
    branch1 = _f32_add(_f32_mul(norm1, w1), b1).to(tl.bfloat16)
    summed = _f32_add(branch0.to(tl.float32), branch1.to(tl.float32)).to(tl.bfloat16)
    relu = tl.maximum(summed.to(tl.float32), 0.0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, relu, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 8e339e83: (T([128,512,1,1], bf16), T([128,512,1,1], bf16), T([512], bf16), ...)
@oracle_impl(hardware="B200", point="8e339e83", BLOCK_M=16, BLOCK_K=16, num_warps=1, num_stages=3)
# d4aaa8d6: (T([128,256,2,2], bf16), T([128,256,2,2], bf16), T([256], bf16), ...)
@oracle_impl(hardware="B200", point="d4aaa8d6", BLOCK_M=8, BLOCK_K=32, num_warps=1, num_stages=3)
# a1bd90d2: (T([128,128,4,4], bf16), T([128,128,4,4], bf16), T([128], bf16), ...)
@oracle_impl(hardware="B200", point="a1bd90d2", BLOCK_M=4, BLOCK_K=64, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0, _shape1, shape2, _shape3 = inputs
    group_shape = _as_shape(shape0)
    out_shape = _as_shape(shape2)
    batch = int(out_shape[0])
    channels = int(out_shape[1])
    height = int(out_shape[2])
    width = int(out_shape[3])
    num_groups = int(group_shape[1])
    group_elems = int(group_shape[2]) * int(group_shape[3])
    total_groups = batch * num_groups
    hw_size = height * width

    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _dual_groupnorm_affine_relu_kernel[(triton.cdiv(total_groups, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        TOTAL_GROUPS=total_groups,
        CHANNELS=channels,
        HW_SIZE=hw_size,
        GROUP_ELEMS=group_elems,
        NUM_GROUPS=num_groups,
        EPS=1.0e-5,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
