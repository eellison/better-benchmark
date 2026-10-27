"""cuTile port of var_mean_54fd2dc43b14: GroupNorm+ReLU+maxpool-with-offsets.

One cuTile kernel per batch*group row computes GroupNorm stats and produces
the affine+ReLU + 3x3 stride-2 maxpool outputs (with earliest-tie offsets).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NUM_GROUPS = 32
EPS = 1.0e-5
POOL_KERNEL = 3
POOL_STRIDE = 2
POOL_PADDING = 1
POOL_DILATION = 1


@ct.kernel
def _groupnorm_relu_maxpool_kernel(
    x_ptr,             # f32 flat [B*NUM_GROUPS, GROUP_ELEMS]
    weight_ptr,        # f32 [C]
    bias_ptr,          # f32 [C]
    mean_ptr,          # f32 [B*NUM_GROUPS]
    rsqrt_ptr,         # f32 [B*NUM_GROUPS]
    values_ptr,        # f32 flat [B*NUM_GROUPS, OUT_GROUP_ELEMS]
    offsets_ptr,       # i8 flat [B*NUM_GROUPS, OUT_GROUP_ELEMS]
    values_bf16_ptr,   # bf16 flat [B*NUM_GROUPS, OUT_GROUP_ELEMS]
    NUM_GROUPS_C: ct.Constant[int],
    GROUP_CHANNELS: ct.Constant[int],
    HEIGHT: ct.Constant[int],
    WIDTH: ct.Constant[int],
    OUT_HEIGHT: ct.Constant[int],
    OUT_WIDTH: ct.Constant[int],
    GROUP_ELEMS: ct.Constant[int],
    OUT_GROUP_ELEMS: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_OUT: ct.Constant[int],
):
    group_row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(group_row, 0), shape=(1, BLOCK_K))
    x = ct.astype(x_bf, ct.float32)
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    active_k = ct.reshape(k_idx < GROUP_ELEMS, (1, BLOCK_K))
    x_for_sum = ct.where(active_k, x, 0.0)
    mean_row = ct.sum(x_for_sum) * (1.0 / GROUP_ELEMS)
    centered = ct.where(active_k, x - mean_row, 0.0)
    variance = ct.sum(centered * centered) * (1.0 / GROUP_ELEMS)
    invstd = ct.rsqrt(variance + EPS)
    ct.store(mean_ptr, index=(group_row,), tile=ct.reshape(mean_row, (1,)))
    ct.store(rsqrt_ptr, index=(group_row,), tile=ct.reshape(invstd, (1,)))

    # Compute maxpool over BLOCK_OUT outputs
    out_offsets = ct.arange(BLOCK_OUT, dtype=ct.int32)
    active_out = out_offsets < OUT_GROUP_ELEMS
    out_hw = OUT_HEIGHT * OUT_WIDTH
    in_hw = HEIGHT * WIDTH
    channel_in_group = out_offsets // out_hw
    out_spatial = out_offsets - channel_in_group * out_hw
    out_h = out_spatial // OUT_WIDTH
    out_w = out_spatial - out_h * OUT_WIDTH
    group_id = group_row % NUM_GROUPS_C
    channel = group_id * GROUP_CHANNELS + channel_in_group

    scale = ct.gather(weight_ptr, channel)
    shift = ct.gather(bias_ptr, channel)

    best = ct.full((BLOCK_OUT,), -float("inf"), dtype=ct.float32)
    best_offset = ct.zeros((BLOCK_OUT,), dtype=ct.int32)

    for kh in range(0, 3):
        in_h = out_h * 2 + kh - 1
        valid_h = (in_h >= 0) & (in_h < HEIGHT)
        for kw in range(0, 3):
            in_w = out_w * 2 + kw - 1
            valid = active_out & valid_h & (in_w >= 0) & (in_w < WIDTH)
            safe_local = channel_in_group * in_hw + in_h * WIDTH + in_w
            safe_local = ct.where(valid, safe_local, 0)
            row_idx = ct.full((BLOCK_OUT,), group_row, dtype=ct.int32)
            candidate_bf = ct.gather(x_ptr, (row_idx, safe_local))
            candidate_r = ct.astype(candidate_bf, ct.float32)
            affine = (candidate_r - mean_row) * invstd * scale + shift
            is_nan = affine != affine
            positive = affine > 0.0
            relu = ct.where(positive | is_nan, affine, 0.0)

            local_pool_offset = kh * 3 + kw
            candidate_is_nan = relu != relu
            best_is_nan = best != best
            equal = (relu == best) | (candidate_is_nan & best_is_nan)
            better = (relu > best) | (candidate_is_nan & ~best_is_nan)
            earlier_tie = equal & (local_pool_offset < best_offset)
            take = valid & (better | earlier_tie)
            best = ct.where(take, relu, best)
            best_offset = ct.where(take, local_pool_offset, best_offset)

    ct.scatter(values_ptr,
               ct.reshape(group_row * OUT_GROUP_ELEMS + out_offsets, (BLOCK_OUT,)),
               best, mask=active_out)
    ct.scatter(offsets_ptr,
               ct.reshape(group_row * OUT_GROUP_ELEMS + out_offsets, (BLOCK_OUT,)),
               ct.astype(best_offset, ct.int8), mask=active_out)
    ct.scatter(values_bf16_ptr,
               ct.reshape(group_row * OUT_GROUP_ELEMS + out_offsets, (BLOCK_OUT,)),
               ct.astype(best, ct.bfloat16), mask=active_out)


@oracle_impl(hardware="B200", point="cb45ad26", BLOCK_K=512, BLOCK_OUT=128)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_OUT: int):
    x, weight, bias, _s0, _s1, _pk, _ps = inputs
    device = x.device
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

    mean = torch.empty_strided(stats_shape, stats_stride, device=device, dtype=torch.float32)
    rsqrt = torch.empty_strided(stats_shape, stats_stride, device=device, dtype=torch.float32)
    values = torch.empty_strided(output_shape, output_stride, device=device, dtype=torch.float32)
    offsets = torch.empty_strided(output_shape, output_stride, device=device, dtype=torch.int8)
    values_bf16 = torch.empty_strided(output_shape, output_stride, device=device, dtype=torch.bfloat16)

    x_flat = x.contiguous().view(total_groups, group_elems)
    mean_flat = mean.view(total_groups)
    rsqrt_flat = rsqrt.view(total_groups)
    values_flat = values.view(-1)
    offsets_flat = offsets.view(-1)
    values_bf16_flat = values_bf16.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (total_groups, 1, 1), _groupnorm_relu_maxpool_kernel,
        (x_flat, weight, bias, mean_flat, rsqrt_flat,
         values_flat, offsets_flat, values_bf16_flat,
         NUM_GROUPS, group_channels, height, width,
         out_height, out_width, group_elems, out_group_elems,
         BLOCK_K, BLOCK_OUT),
    )
    return mean, rsqrt, values, offsets, values_bf16
