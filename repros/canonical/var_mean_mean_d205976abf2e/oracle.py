"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GhostNet training-BatchNorm-ReLU-cat-spatial-mean scope by reducing per-channel BatchNorm statistics from the channels-last input, updating both running-stat copy_ outputs, materializing the exact bf16 channel concat, and computing the returned bf16 spatial mean from the post-cast ReLU values, whereas Inductor currently schedules the BatchNorm statistics/update region, affine ReLU, concatenation, and following spatial mean as generic fused regions around large intermediate tensors; Inductor cannot do this today because its norm-template scheduler does not preserve this producer/consumer chain across mutable BatchNorm side outputs and a downstream concat mean; the fix is SCHEDULER_FUSION: teach the BN-training template to expose running-stat side effects while fusing fixed-shape affine/ReLU consumers and logical channel-concat spatial reductions into one planned schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _partial_stats_kernel(
    x_ptr,
    partial_ptr,
    CHANNELS: tl.constexpr,
    HW_SIZE: tl.constexpr,
    ELEMENTS_PER_CHANNEL: tl.constexpr,
    STAT_BLOCKS: tl.constexpr,
    X_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    r_offsets = tl.program_id(1) * R_BLOCK + tl.arange(0, R_BLOCK)[:, None]
    c_offsets = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[None, :]
    mask = (c_offsets < CHANNELS) & (r_offsets < ELEMENTS_PER_CHANNEL)

    n_idx = r_offsets // HW_SIZE
    hw_idx = r_offsets - n_idx * HW_SIZE
    x_offsets = n_idx * CHANNELS * HW_SIZE + hw_idx * CHANNELS + c_offsets
    vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    sums = tl.sum(vals, axis=0)
    sums_sq = tl.sum(vals * vals, axis=0)
    c_1d = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
    out_offsets = c_1d * STAT_BLOCKS + tl.program_id(1)
    c_mask = c_1d < CHANNELS
    tl.store(partial_ptr + out_offsets, sums, mask=c_mask)
    tl.store(partial_ptr + CHANNELS * STAT_BLOCKS + out_offsets, sums_sq, mask=c_mask)


@triton.jit
def _finalize_stats_kernel(
    partial_ptr,
    running_mean_ptr,
    running_var_ptr,
    mean_out_ptr,
    invstd_out_ptr,
    CHANNELS: tl.constexpr,
    ELEMENTS_PER_CHANNEL: tl.constexpr,
    STAT_BLOCKS: tl.constexpr,
    FINAL_X_BLOCK: tl.constexpr,
    FINAL_BLOCK: tl.constexpr,
):
    p_offsets = tl.arange(0, FINAL_BLOCK)[:, None]
    c_offsets = tl.program_id(0) * FINAL_X_BLOCK + tl.arange(0, FINAL_X_BLOCK)[None, :]
    mask = (c_offsets < CHANNELS) & (p_offsets < STAT_BLOCKS)
    partial_offsets = c_offsets * STAT_BLOCKS + p_offsets

    sums = tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
    sums_sq = tl.load(
        partial_ptr + CHANNELS * STAT_BLOCKS + partial_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total_sum = tl.sum(sums, axis=0)
    total_sum_sq = tl.sum(sums_sq, axis=0)
    mean = total_sum / ELEMENTS_PER_CHANNEL
    var = total_sum_sq / ELEMENTS_PER_CHANNEL - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + 1.0e-5)

    c_1d = tl.program_id(0) * FINAL_X_BLOCK + tl.arange(0, FINAL_X_BLOCK)
    c_mask = c_1d < CHANNELS
    old_mean = tl.load(running_mean_ptr + c_1d, mask=c_mask, other=0.0).to(tl.float32)
    old_var = tl.load(running_var_ptr + c_1d, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(running_mean_ptr + c_1d, old_mean * 0.9 + mean * 0.1, mask=c_mask)
    tl.store(
        running_var_ptr + c_1d,
        old_var * 0.9 + var * 1.0000398612827361 * 0.1,
        mask=c_mask,
    )
    tl.store(mean_out_ptr + c_1d, mean, mask=c_mask)
    tl.store(invstd_out_ptr + c_1d, invstd, mask=c_mask)


@triton.jit
def _cat_and_spatial_mean_kernel(
    x_ptr,
    skip_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    cat_ptr,
    spatial_mean_ptr,
    CHANNELS: tl.constexpr,
    HW_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    hw_offsets = tl.arange(0, BLOCK_HW)[:, None]
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    hw_mask = hw_offsets < HW_SIZE
    c_mask = c_offsets < CHANNELS
    mask = hw_mask & c_mask

    in_offsets = n * CHANNELS * HW_SIZE + hw_offsets * CHANNELS + c_offsets
    cat_channels = CHANNELS * 2
    cat_base = n * cat_channels * HW_SIZE + hw_offsets * cat_channels

    skip = tl.load(skip_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(cat_ptr + cat_base + c_offsets, skip, mask=mask)
    skip_f32 = tl.where(hw_mask, skip.to(tl.float32), 0.0)
    skip_mean = tl.sum(skip_f32, axis=0) / HW_SIZE

    x = tl.load(x_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    y = (x - mean) * invstd
    y = y * weight + bias
    y_bf16 = y.to(tl.bfloat16).to(tl.float32)
    relu = tl.where(y_bf16 != y_bf16, y_bf16, tl.maximum(y_bf16, 0.0))
    relu = tl.where(hw_mask, relu, 0.0)
    tl.store(cat_ptr + cat_base + CHANNELS + c_offsets, relu.to(tl.bfloat16), mask=mask)
    relu_mean = tl.sum(relu, axis=0) / HW_SIZE

    c_1d = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_1d_mask = c_1d < CHANNELS
    mean_base = n * cat_channels
    tl.store(spatial_mean_ptr + mean_base + c_1d, skip_mean.to(tl.bfloat16), mask=c_1d_mask)
    tl.store(
        spatial_mean_ptr + mean_base + CHANNELS + c_1d,
        relu_mean.to(tl.bfloat16),
        mask=c_1d_mask,
    )


# d3838873: (T([512,480,7,7], bf16), T([480], f32), T([480], f32), T([480], f32), T([480], f32), T([512,480,7,7], bf16))
@oracle_impl(hardware="B200", point="d3838873", STAT_X=16, STAT_BLOCK=512, FINAL_X=16, FINAL_BLOCK=64, BLOCK_C=16, BLOCK_HW=64, stat_warps=8, final_warps=1, output_warps=4)
# c9e11f4c: (T([512,336,14,14], bf16), T([336], f32), T([336], f32), T([336], f32), T([336], f32), T([512,336,14,14], bf16))
@oracle_impl(hardware="B200", point="c9e11f4c", STAT_X=16, STAT_BLOCK=512, FINAL_X=16, FINAL_BLOCK=256, BLOCK_C=16, BLOCK_HW=256, stat_warps=8, final_warps=4, output_warps=8)
# 0979feae: (T([512,240,14,14], bf16), T([240], f32), T([240], f32), T([240], f32), T([240], f32), T([512,240,14,14], bf16))
@oracle_impl(hardware="B200", point="0979feae", STAT_X=16, STAT_BLOCK=512, FINAL_X=16, FINAL_BLOCK=256, BLOCK_C=16, BLOCK_HW=256, stat_warps=8, final_warps=4, output_warps=8)
# a58059f1: (T([512,60,28,28], bf16), T([60], f32), T([60], f32), T([60], f32), T([60], f32), T([512,60,28,28], bf16))
@oracle_impl(hardware="B200", point="a58059f1", STAT_X=16, STAT_BLOCK=512, FINAL_X=16, FINAL_BLOCK=1024, BLOCK_C=32, BLOCK_HW=1024, stat_warps=8, final_warps=8, output_warps=8)
def oracle_forward(
    inputs,
    *,
    STAT_X: int,
    STAT_BLOCK: int,
    FINAL_X: int,
    FINAL_BLOCK: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    stat_warps: int,
    final_warps: int,
    output_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    n_size = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw_size = height * width
    elements_per_channel = n_size * hw_size
    stat_blocks = triton.cdiv(elements_per_channel, STAT_BLOCK)
    cat_channels = channels * 2

    partial = torch.empty_strided(
        (2, channels, stat_blocks),
        (channels * stat_blocks, stat_blocks, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    mean_out = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    invstd_out = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    cat = torch.empty_strided(
        (n_size, cat_channels, height, width),
        (cat_channels * hw_size, 1, width * cat_channels, cat_channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    spatial_mean = torch.empty_strided(
        (n_size, cat_channels, 1, 1),
        (cat_channels, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _partial_stats_kernel[(triton.cdiv(channels, STAT_X), stat_blocks)](
        arg0_1,
        partial,
        CHANNELS=channels,
        HW_SIZE=hw_size,
        ELEMENTS_PER_CHANNEL=elements_per_channel,
        STAT_BLOCKS=stat_blocks,
        X_BLOCK=STAT_X,
        R_BLOCK=STAT_BLOCK,
        num_warps=stat_warps,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(channels, FINAL_X),)](
        partial,
        arg1_1,
        arg2_1,
        mean_out,
        invstd_out,
        CHANNELS=channels,
        ELEMENTS_PER_CHANNEL=elements_per_channel,
        STAT_BLOCKS=stat_blocks,
        FINAL_X_BLOCK=FINAL_X,
        FINAL_BLOCK=FINAL_BLOCK,
        num_warps=final_warps,
        num_stages=3,
    )
    _cat_and_spatial_mean_kernel[(n_size, triton.cdiv(channels, BLOCK_C))](
        arg0_1,
        arg5_1,
        arg3_1,
        arg4_1,
        mean_out,
        invstd_out,
        cat,
        spatial_mean,
        CHANNELS=channels,
        HW_SIZE=hw_size,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=output_warps,
        num_stages=3,
    )
    return mean_out, invstd_out, cat, spatial_mean, arg1_1, arg2_1
