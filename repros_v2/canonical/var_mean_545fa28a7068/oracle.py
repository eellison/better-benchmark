"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 groupnorm-style var_mean, affine bf16-rounding ReLU, and observed low-memory 3x3 stride-2 maxpool value output in one shape-specialized Triton kernel, whereas Inductor schedules the normalization producer, bf16 activation boundary, and pooling stencil as generic reduction/pointwise/pooling work; Inductor cannot do this today because scheduler fusion does not model a reduction producer feeding a stencil consumer while preserving bf16 cast, ReLU NaN, and low-memory maxpool value semantics; the fix is SCHEDULER_FUSION: add a GroupNorm-affine-ReLU-to-maxpool fusion template that computes group statistics once and lowers the pooled value epilogue directly from the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


NUM_GROUPS = 32
EPS = 1.0e-5
POOL_KERNEL = 3
POOL_STRIDE = 2
POOL_PADDING = 1
POOL_DILATION = 1


@triton.jit
def _groupnorm_bf16_relu_maxpool_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    NUM_GROUPS_CONST: tl.constexpr,
    GROUP_CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_HEIGHT: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    GROUP_ELEMS: tl.constexpr,
    OUT_GROUP_ELEMS: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    group_row = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_K)
    active_k = k_offsets < GROUP_ELEMS
    x_base = group_row * GROUP_ELEMS

    x = tl.load(x_ptr + x_base + k_offsets, mask=active_k, other=0.0).to(tl.float32)
    x_for_sum = tl.where(active_k, x, 0.0)
    mean = tl.sum(x_for_sum, axis=0) / GROUP_ELEMS
    centered = tl.where(active_k, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / GROUP_ELEMS
    invstd = libdevice.rsqrt(variance + EPSILON)

    out_offsets = tl.arange(0, BLOCK_OUT)
    active_out = out_offsets < OUT_GROUP_ELEMS
    out_hw = OUT_HEIGHT * OUT_WIDTH
    channel_in_group = out_offsets // out_hw
    out_spatial = out_offsets - channel_in_group * out_hw
    out_h = out_spatial // OUT_WIDTH
    out_w = out_spatial - out_h * OUT_WIDTH

    group_id = group_row % NUM_GROUPS_CONST
    channel = group_id * GROUP_CHANNELS + channel_in_group
    scale = tl.load(weight_ptr + channel, mask=active_out, other=0.0).to(tl.float32)
    shift = tl.load(bias_ptr + channel, mask=active_out, other=0.0).to(tl.float32)

    best = tl.full((BLOCK_OUT,), -float("inf"), tl.float32)

    for kh in tl.static_range(0, 3):
        in_h = out_h * 2 + kh - 1
        valid_h = (in_h >= 0) & (in_h < HEIGHT)
        for kw in tl.static_range(0, 3):
            in_w = out_w * 2 + kw - 1
            valid = active_out & valid_h & (in_w >= 0) & (in_w < WIDTH)
            local_offset = channel_in_group * (HEIGHT * WIDTH) + in_h * WIDTH + in_w
            candidate = tl.load(
                x_ptr + x_base + local_offset,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            normalized = (candidate - mean) * invstd
            affine = normalized * scale + shift
            rounded = affine.to(tl.bfloat16)
            relu = tl.where((rounded > 0.0) | (rounded != rounded), rounded, 0.0).to(tl.float32)
            take = valid & ((relu > best) | (relu != relu))
            best = tl.where(take, relu, best)

    out_base = group_row * OUT_GROUP_ELEMS
    tl.store(out_ptr + out_base + out_offsets, best.to(tl.bfloat16), mask=active_out)


# 1e6bd5b8: (T([128,64,16,16], bf16), T([64], bf16), T([64], bf16), S([128,32,2,256]), S([128,64,16,16]), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="1e6bd5b8", BLOCK_K=512, BLOCK_OUT=128, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_OUT: int, num_warps: int, num_stages: int):
    x, weight, bias, _group_shape, _activation_shape, _pool_kernel, _pool_stride = inputs
    batch = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    group_channels = channels // NUM_GROUPS
    out_height = (height + 2 * POOL_PADDING - POOL_DILATION * (POOL_KERNEL - 1) - 1) // POOL_STRIDE + 1
    out_width = (width + 2 * POOL_PADDING - POOL_DILATION * (POOL_KERNEL - 1) - 1) // POOL_STRIDE + 1
    group_elems = group_channels * height * width
    out_group_elems = group_channels * out_height * out_width
    total_groups = batch * NUM_GROUPS

    out = torch.empty(
        (batch, channels, out_height, out_width),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _groupnorm_bf16_relu_maxpool_kernel[(total_groups,)](
        x,
        weight,
        bias,
        out,
        NUM_GROUPS_CONST=NUM_GROUPS,
        GROUP_CHANNELS=group_channels,
        HEIGHT=height,
        WIDTH=width,
        OUT_HEIGHT=out_height,
        OUT_WIDTH=out_width,
        GROUP_ELEMS=group_elems,
        OUT_GROUP_ELEMS=out_group_elems,
        EPSILON=EPS,
        BLOCK_K=BLOCK_K,
        BLOCK_OUT=BLOCK_OUT,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
