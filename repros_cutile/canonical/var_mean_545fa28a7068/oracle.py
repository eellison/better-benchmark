"""cuTile port of var_mean_545fa28a7068: GroupNorm + affine + ReLU +
3x3 stride-2 maxpool fused in ONE kernel (mirrors Triton kernel exactly).

Grid: (batch * NUM_GROUPS,) — each program computes GN stats over one group's
tile (GC, H, W), then computes the pooled bf16 output for that group's
(GC, OH, OW) output positions by manual 3x3 stencil load + affine + ReLU.

Uses ct.gather over the flat group tile to grab the 9 stencil positions with
per-element bounds masking (mirrors Triton's per-load mask).
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
def _gn_relu_maxpool_kernel(
    x_ptr,          # bf16 [TOTAL_GROUPS * GROUP_ELEMS]  flat
    weight_ptr,     # bf16 [CHANNELS]
    bias_ptr,       # bf16 [CHANNELS]
    out_ptr,        # bf16 [TOTAL_GROUPS * OUT_GROUP_ELEMS] flat
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
    group_row = ct.bid(0)  # 0..TOTAL_GROUPS
    # ---- load full group tile as flat [BLOCK_K] ----
    k_offsets = ct.arange(BLOCK_K, dtype=ct.int32)
    active_k = k_offsets < GROUP_ELEMS
    x_base = group_row * GROUP_ELEMS
    # gather over the flat x buffer
    x_indices = x_base + k_offsets
    x = ct.astype(ct.gather(x_ptr, x_indices), ct.float32)
    zero_1d = ct.zeros((BLOCK_K,), dtype=ct.float32)
    x_for_sum = ct.where(active_k, x, zero_1d)
    mean = ct.sum(x_for_sum) * (1.0 / GROUP_ELEMS)
    centered = ct.where(active_k, x - mean, zero_1d)
    variance = ct.sum(centered * centered) * (1.0 / GROUP_ELEMS)
    invstd = ct.rsqrt(variance + EPS)

    # ---- 3x3 stride-2 maxpool with padding=1 ----
    out_offsets = ct.arange(BLOCK_OUT, dtype=ct.int32)
    active_out = out_offsets < OUT_GROUP_ELEMS
    out_hw = OUT_HEIGHT * OUT_WIDTH
    channel_in_group = out_offsets // out_hw
    out_spatial = out_offsets - channel_in_group * out_hw
    out_h = out_spatial // OUT_WIDTH
    out_w = out_spatial - out_h * OUT_WIDTH

    group_id = group_row - (group_row // NUM_GROUPS_C) * NUM_GROUPS_C
    channel = group_id * GROUP_CHANNELS + channel_in_group
    scale = ct.astype(ct.gather(weight_ptr, channel), ct.float32)
    shift = ct.astype(ct.gather(bias_ptr, channel), ct.float32)

    zero_out = ct.zeros((BLOCK_OUT,), dtype=ct.float32)
    neg_inf = ct.full((BLOCK_OUT,), -3.4e38, dtype=ct.float32)
    best = neg_inf

    for kh in range(3):
        in_h = out_h * 2 + kh - 1
        valid_h = (in_h >= 0) & (in_h < HEIGHT)
        for kw in range(3):
            in_w = out_w * 2 + kw - 1
            valid = active_out & valid_h & (in_w >= 0) & (in_w < WIDTH)
            local_offset = channel_in_group * (HEIGHT * WIDTH) + in_h * WIDTH + in_w
            # Clamp local_offset to a safe value when invalid so gather stays in-bounds.
            safe_local = ct.where(valid, local_offset, ct.zeros((BLOCK_OUT,), dtype=ct.int32))
            candidate = ct.astype(ct.gather(x_ptr, x_base + safe_local), ct.float32)
            normalized = (candidate - mean) * invstd
            affine = normalized * scale + shift
            rounded = ct.astype(affine, ct.bfloat16)
            rounded_f = ct.astype(rounded, ct.float32)
            is_nan = rounded_f != rounded_f
            take_orig = (rounded_f > 0.0) | is_nan
            relu = ct.where(take_orig, rounded_f, zero_out)
            # NaN-aware max: NaN wins.
            relu_is_nan = relu != relu
            take = valid & ((relu > best) | relu_is_nan)
            best = ct.where(take, relu, best)

    # BLOCK_OUT == OUT_GROUP_ELEMS so we can store the row as a (1, BLOCK_OUT) tile.
    result_bf = ct.astype(best, ct.bfloat16)
    ct.store(out_ptr, index=(group_row, 0), tile=ct.reshape(result_bf, (1, BLOCK_OUT)))


# 1e6bd5b8: (T([128,64,16,16], bf16), T([64], bf16), T([64], bf16), S([128,32,2,256]), S([128,64,16,16]), S([3,3]), S([2,2]))
@oracle_impl(hardware="B200", point="1e6bd5b8", BLOCK_K=512, BLOCK_OUT=128)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_OUT: int):
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

    out = torch.empty(
        (batch, channels, out_height, out_width),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Flat views used inside the kernel.
    x_flat = x.reshape(total_groups * group_elems)
    out_flat = out.view(total_groups, out_group_elems)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_groups, 1, 1),
        _gn_relu_maxpool_kernel,
        (x_flat, weight, bias, out_flat,
         NUM_GROUPS, group_channels, height, width, out_height, out_width,
         group_elems, out_group_elems, BLOCK_K, BLOCK_OUT),
    )
    return out
