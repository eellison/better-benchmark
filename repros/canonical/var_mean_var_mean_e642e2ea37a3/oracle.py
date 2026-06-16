"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 ShuffleNet dual training-BatchNorm plus channel-shuffle scope, including both fp32 population var_mean reductions over N,H,W, both eps=1e-5 rsqrt and mean side outputs, four mutable running-stat copy_ aliases with the captured unbiased-variance correction, the two affine bf16 cast and ReLU boundaries, and the cat/view/permute/clone/view/split layout by writing the final shuffled storage directly, whereas Inductor lowers the sibling BN-training reductions, mutable copy_ epilogues, affine/ReLU branches, channel cat, layout clone, and split views as generic schedules; Inductor cannot do this today because its normalization scheduler does not co-schedule paired mutable BN reductions with a fixed channel-shuffle layout consumer while preserving bf16 rounding boundaries, aliases, and returned split strides; the fix is SCHEDULER_FUSION: add a guarded dual-training-BN plus channel-shuffle lowering that groups sibling channel statistics, running-stat side effects, and direct shuffled-layout materialization in one full-scope plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
OLD_WEIGHT = 0.9
RUNNING_VAR_CORRECTION = 1.0001594642002871


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
def _dual_bn_stats_kernel(
    x0_ptr,
    x1_ptr,
    running_mean0_ptr,
    running_var0_ptr,
    running_mean1_ptr,
    running_var1_ptr,
    mean0_ptr,
    rsqrt0_ptr,
    mean1_ptr,
    rsqrt1_ptr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    elements_per_channel: tl.constexpr,
    x0_stride_n: tl.constexpr,
    x0_stride_c: tl.constexpr,
    x0_stride_h: tl.constexpr,
    x0_stride_w: tl.constexpr,
    x1_stride_n: tl.constexpr,
    x1_stride_c: tl.constexpr,
    x1_stride_h: tl.constexpr,
    x1_stride_w: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    old_weight: tl.constexpr,
    running_var_correction: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channels_vec = tl.program_id(0) + tl.arange(0, 1)
    r_offsets = tl.arange(0, BLOCK_R)
    r_mask = r_offsets < elements_per_channel
    c_mask = channels_vec < channels
    mask = c_mask[:, None] & r_mask[None, :]
    hw_size: tl.constexpr = height * width
    n_idx = r_offsets // hw_size
    hw_idx = r_offsets - n_idx * hw_size
    h_idx = hw_idx // width
    w_idx = hw_idx - h_idx * width

    x0_offsets = (
        n_idx[None, :] * x0_stride_n
        + channels_vec[:, None] * x0_stride_c
        + h_idx[None, :] * x0_stride_h
        + w_idx[None, :] * x0_stride_w
    )
    x1_offsets = (
        n_idx[None, :] * x1_stride_n
        + channels_vec[:, None] * x1_stride_c
        + h_idx[None, :] * x1_stride_h
        + w_idx[None, :] * x1_stride_w
    )
    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)

    mean_acc0 = tl.zeros((1, BLOCK_R), tl.float32)
    m2_acc0 = tl.zeros((1, BLOCK_R), tl.float32)
    weight_acc0 = tl.zeros((1, BLOCK_R), tl.float32)
    mean_next0, m2_next0, weight_next0 = triton_helpers.welford_reduce(
        x0,
        mean_acc0,
        m2_acc0,
        weight_acc0,
        True,
    )
    mean_acc0 = tl.where(mask, mean_next0, mean_acc0)
    m2_acc0 = tl.where(mask, m2_next0, m2_acc0)
    weight_acc0 = tl.where(mask, weight_next0, weight_acc0)
    mean0, m20, _weight0 = triton_helpers.welford(
        mean_acc0,
        m2_acc0,
        weight_acc0,
        1,
    )

    mean_acc1 = tl.zeros((1, BLOCK_R), tl.float32)
    m2_acc1 = tl.zeros((1, BLOCK_R), tl.float32)
    weight_acc1 = tl.zeros((1, BLOCK_R), tl.float32)
    mean_next1, m2_next1, weight_next1 = triton_helpers.welford_reduce(
        x1,
        mean_acc1,
        m2_acc1,
        weight_acc1,
        True,
    )
    mean_acc1 = tl.where(mask, mean_next1, mean_acc1)
    m2_acc1 = tl.where(mask, m2_next1, m2_acc1)
    weight_acc1 = tl.where(mask, weight_next1, weight_acc1)
    mean1, m21, _weight1 = triton_helpers.welford(
        mean_acc1,
        m2_acc1,
        weight_acc1,
        1,
    )

    inv_count: tl.constexpr = 1.0 / elements_per_channel
    var0 = _f32_mul(m20, inv_count)
    var1 = _f32_mul(m21, inv_count)
    rsqrt0 = libdevice.rsqrt(_f32_add(var0, eps))
    rsqrt1 = libdevice.rsqrt(_f32_add(var1, eps))

    old_mean0 = tl.load(running_mean0_ptr + channels_vec, mask=c_mask, other=0.0).to(tl.float32)
    old_var0 = tl.load(running_var0_ptr + channels_vec, mask=c_mask, other=0.0).to(tl.float32)
    old_mean1 = tl.load(running_mean1_ptr + channels_vec, mask=c_mask, other=0.0).to(tl.float32)
    old_var1 = tl.load(running_var1_ptr + channels_vec, mask=c_mask, other=0.0).to(tl.float32)
    new_mean0 = _f32_add(_f32_mul(mean0, momentum), _f32_mul(old_mean0, old_weight))
    new_var0 = _f32_add(
        _f32_mul(_f32_mul(var0, running_var_correction), momentum),
        _f32_mul(old_var0, old_weight),
    )
    new_mean1 = _f32_add(_f32_mul(mean1, momentum), _f32_mul(old_mean1, old_weight))
    new_var1 = _f32_add(
        _f32_mul(_f32_mul(var1, running_var_correction), momentum),
        _f32_mul(old_var1, old_weight),
    )

    tl.store(mean0_ptr + channels_vec, mean0, mask=c_mask)
    tl.store(rsqrt0_ptr + channels_vec, rsqrt0, mask=c_mask)
    tl.store(mean1_ptr + channels_vec, mean1, mask=c_mask)
    tl.store(rsqrt1_ptr + channels_vec, rsqrt1, mask=c_mask)
    tl.store(running_mean0_ptr + channels_vec, new_mean0, mask=c_mask)
    tl.store(running_var0_ptr + channels_vec, new_var0, mask=c_mask)
    tl.store(running_mean1_ptr + channels_vec, new_mean1, mask=c_mask)
    tl.store(running_var1_ptr + channels_vec, new_var1, mask=c_mask)


@triton.jit
def _dual_bn_shuffle_kernel(
    x0_ptr,
    x1_ptr,
    weight0_ptr,
    bias0_ptr,
    weight1_ptr,
    bias1_ptr,
    mean0_ptr,
    rsqrt0_ptr,
    mean1_ptr,
    rsqrt1_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    x0_stride_n: tl.constexpr,
    x0_stride_c: tl.constexpr,
    x0_stride_h: tl.constexpr,
    x0_stride_w: tl.constexpr,
    x1_stride_n: tl.constexpr,
    x1_stride_c: tl.constexpr,
    x1_stride_h: tl.constexpr,
    x1_stride_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    hw_size: tl.constexpr = height * width
    hw_idx = offsets % hw_size
    channel = (offsets // hw_size) % channels
    n_idx = offsets // (channels * hw_size)
    h_idx = hw_idx // width
    w_idx = hw_idx - h_idx * width

    x0_offsets = (
        n_idx * x0_stride_n
        + channel * x0_stride_c
        + h_idx * x0_stride_h
        + w_idx * x0_stride_w
    )
    x1_offsets = (
        n_idx * x1_stride_n
        + channel * x1_stride_c
        + h_idx * x1_stride_h
        + w_idx * x1_stride_w
    )
    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    rsqrt0 = tl.load(rsqrt0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    rsqrt1 = tl.load(rsqrt1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    y0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x0, mean0), rsqrt0), weight0), bias0)
    y1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(x1, mean1), rsqrt1), weight1), bias1)
    y0_bf16 = y0.to(tl.bfloat16, fp_downcast_rounding="rtne")
    y1_bf16 = y1.to(tl.bfloat16, fp_downcast_rounding="rtne")
    relu0 = tl.where((y0_bf16 > 0.0) | (y0_bf16 != y0_bf16), y0_bf16, 0.0)
    relu1 = tl.where((y1_bf16 > 0.0) | (y1_bf16 != y1_bf16), y1_bf16, 0.0)

    out_hw = hw_idx
    out_base = n_idx * (channels * 2 * hw_size) + channel * (2 * hw_size) + out_hw
    tl.store(out_ptr + out_base, relu0, mask=mask)
    tl.store(out_ptr + out_base + hw_size, relu1, mask=mask)


@oracle_impl(hardware="B200", point="24b29630", BLOCK_R=8192, BLOCK=256, num_warps_stats=8, num_warps_act=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK: int,
    num_warps_stats: int,
    num_warps_act: int,
):
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        shape0,
        shape1,
    ) = inputs
    del shape0

    batch = int(x0.shape[0])
    channels = int(x0.shape[1])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    hw_size = height * width
    elements_per_channel = batch * hw_size
    total_elements = batch * channels * hw_size
    shuffled_shape = tuple(int(dim) for dim in shape1)

    mean0 = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    mean1 = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    rsqrt0 = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    rsqrt1 = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    shuffled = torch.empty_strided(
        shuffled_shape,
        (shuffled_shape[1] * hw_size, hw_size, width, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    _dual_bn_stats_kernel[(channels,)](
        x0,
        x1,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        mean0,
        rsqrt0,
        mean1,
        rsqrt1,
        channels=channels,
        height=height,
        width=width,
        elements_per_channel=elements_per_channel,
        x0_stride_n=int(x0.stride(0)),
        x0_stride_c=int(x0.stride(1)),
        x0_stride_h=int(x0.stride(2)),
        x0_stride_w=int(x0.stride(3)),
        x1_stride_n=int(x1.stride(0)),
        x1_stride_c=int(x1.stride(1)),
        x1_stride_h=int(x1.stride(2)),
        x1_stride_w=int(x1.stride(3)),
        eps=EPS,
        momentum=MOMENTUM,
        old_weight=OLD_WEIGHT,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps_stats,
        num_stages=3,
    )
    _dual_bn_shuffle_kernel[(triton.cdiv(total_elements, BLOCK),)](
        x0,
        x1,
        weight0,
        bias0,
        weight1,
        bias1,
        mean0,
        rsqrt0,
        mean1,
        rsqrt1,
        shuffled,
        total_elements=total_elements,
        channels=channels,
        height=height,
        width=width,
        x0_stride_n=int(x0.stride(0)),
        x0_stride_c=int(x0.stride(1)),
        x0_stride_h=int(x0.stride(2)),
        x0_stride_w=int(x0.stride(3)),
        x1_stride_n=int(x1.stride(0)),
        x1_stride_c=int(x1.stride(1)),
        x1_stride_h=int(x1.stride(2)),
        x1_stride_w=int(x1.stride(3)),
        BLOCK=BLOCK,
        num_warps=num_warps_act,
        num_stages=3,
    )

    first, second = torch.split(shuffled, channels, dim=1)
    return (
        mean0,
        rsqrt0,
        mean1,
        rsqrt1,
        first,
        second,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )
