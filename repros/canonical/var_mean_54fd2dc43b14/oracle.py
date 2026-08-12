"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16-input groupnorm-style var_mean, f32 affine ReLU, low-memory 3x3 stride-2 maxpool-with-offsets, and final bf16 pooled-value cast in one shape-specialized Triton kernel while returning the exposed mean and rsqrt side outputs, whereas Inductor materializes the full f32 normalized ReLU activation between a generic reduction kernel and a separate pooling stencil; Inductor cannot do this today because scheduler fusion does not model a reduction-backed normalization producer feeding a multi-output low-memory maxpool stencil while preserving exposed statistics, f32 ReLU, offset tie/NaN semantics, and the final bf16 cast; the fix is SCHEDULER_FUSION: add a GroupNorm-affine-ReLU-to-low-memory-maxpool fusion template that computes group statistics once and emits stats, pooled values, offsets, and casted pooled values from the same full-scope schedule."""

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
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _groupnorm_relu_maxpool_full_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rsqrt_ptr,
    values_ptr,
    offsets_ptr,
    values_bf16_ptr,
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

    tl.store(mean_ptr + group_row, mean)
    tl.store(rsqrt_ptr + group_row, invstd)

    out_offsets = tl.arange(0, BLOCK_OUT)
    active_out = out_offsets < OUT_GROUP_ELEMS
    out_hw = OUT_HEIGHT * OUT_WIDTH
    in_hw = HEIGHT * WIDTH
    channel_in_group = out_offsets // out_hw
    out_spatial = out_offsets - channel_in_group * out_hw
    out_h = out_spatial // OUT_WIDTH
    out_w = out_spatial - out_h * OUT_WIDTH

    group_id = group_row % NUM_GROUPS_CONST
    channel = group_id * GROUP_CHANNELS + channel_in_group
    scale = tl.load(weight_ptr + channel, mask=active_out, other=0.0).to(tl.float32)
    shift = tl.load(bias_ptr + channel, mask=active_out, other=0.0).to(tl.float32)

    best = tl.full((BLOCK_OUT,), -float("inf"), tl.float32)
    best_offset = tl.zeros((BLOCK_OUT,), tl.int32)

    for kh in tl.static_range(0, 3):
        in_h = out_h * 2 + kh - 1
        valid_h = (in_h >= 0) & (in_h < HEIGHT)
        for kw in tl.static_range(0, 3):
            in_w = out_w * 2 + kw - 1
            valid = active_out & valid_h & (in_w >= 0) & (in_w < WIDTH)
            local_input = channel_in_group * in_hw + in_h * WIDTH + in_w
            candidate = tl.load(
                x_ptr + x_base + local_input,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            affine = ((candidate - mean) * invstd) * scale + shift
            relu = _relu_preserve_nan(affine)

            local_pool_offset = kh * 3 + kw
            candidate_is_nan = relu != relu
            best_is_nan = best != best
            equal = (relu == best) | (candidate_is_nan & best_is_nan)
            better = (relu > best) | (candidate_is_nan & ~best_is_nan)
            earlier_tie = equal & (local_pool_offset < best_offset)
            take = valid & (better | earlier_tie)
            best = tl.where(take, relu, best)
            best_offset = tl.where(take, local_pool_offset, best_offset)

    out_base = group_row * OUT_GROUP_ELEMS
    store_offsets = out_base + out_offsets
    tl.store(values_ptr + store_offsets, best, mask=active_out)
    tl.store(offsets_ptr + store_offsets, best_offset.to(tl.int8), mask=active_out)
    tl.store(values_bf16_ptr + store_offsets, best, mask=active_out)


# cb45ad26: (T([128,64,16,16], bf16), T([64], f32), T([64], f32), S([128,32,2,256]), S([128,64,16,16]), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="cb45ad26", BLOCK_K=512, BLOCK_OUT=128, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_OUT: int,
    num_warps: int,
    num_stages: int,
):
    x, weight, bias, _group_shape, _activation_shape, _pool_kernel, _pool_stride = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    group_channels = channels // NUM_GROUPS
    out_height = (height + 2 * POOL_PADDING - POOL_DILATION * (POOL_KERNEL - 1) - 1) // POOL_STRIDE + 1
    out_width = (width + 2 * POOL_PADDING - POOL_DILATION * (POOL_KERNEL - 1) - 1) // POOL_STRIDE + 1
    group_elems = group_channels * height * width
    out_group_elems = group_channels * out_height * out_width
    total_groups = batch * NUM_GROUPS

    stats_shape = (batch, NUM_GROUPS, 1, 1)
    stats_stride = (NUM_GROUPS, 1, 1, 1)
    output_shape = (batch, channels, out_height, out_width)
    output_stride = (channels * out_height * out_width, out_height * out_width, out_width, 1)

    mean = torch.empty_strided(stats_shape, stats_stride, device=x.device, dtype=torch.float32)
    rsqrt = torch.empty_strided(stats_shape, stats_stride, device=x.device, dtype=torch.float32)
    values = torch.empty_strided(output_shape, output_stride, device=x.device, dtype=torch.float32)
    offsets = torch.empty_strided(output_shape, output_stride, device=x.device, dtype=torch.int8)
    values_bf16 = torch.empty_strided(output_shape, output_stride, device=x.device, dtype=torch.bfloat16)

    _groupnorm_relu_maxpool_full_kernel[(total_groups,)](
        x,
        weight,
        bias,
        mean,
        rsqrt,
        values,
        offsets,
        values_bf16,
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
    return mean, rsqrt, values, offsets, values_bf16
