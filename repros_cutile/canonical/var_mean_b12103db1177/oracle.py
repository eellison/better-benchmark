"""cuTile port of var_mean_b12103db1177: ShuffleNet BN training + channel-shuffle skip.

Approach:
  1. `_bn_stats_kernel`: per-channel fp64 population var_mean over N*H*W,
     stores saved mean and invstd. Gather x into a `[C, N*H*W]` layout so
     each channel is one row.
  2. `_bn_affine_relu_kernel`: per-element BN affine (fp64), bf16 cast + ReLU.
     Uses 2D indexing (row = tile idx, col = position-in-tile) with padding
     because `ct.load(index=(runtime_scalar,), shape=(B,))` on 1D tensors
     was observed to miscompile at large launch grids.
  3. torch epilogue: interleaved channel-shuffle store + running-stat update.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871

BLOCK_K = 8192
BLOCK_E = 1024


@ct.kernel
def _bn_stats_from_gathered_kernel(
    xg_ptr,          # bf16 [C, E] gathered per-channel
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    E_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
):
    channel = ct.bid(0)
    xg = ct.load(
        xg_ptr, index=(channel, 0), shape=(1, BLOCK_K_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x64 = ct.astype(xg, ct.float64)
    col_idx = ct.arange(BLOCK_K_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < E_, (1, BLOCK_K_))
    x64_masked = ct.where(col_mask, x64, 0.0)
    total = ct.sum(x64_masked)
    mean64 = total * (1.0 / E_)
    centered64 = ct.where(col_mask, x64 - mean64, 0.0)
    var64 = ct.sum(centered64 * centered64) * (1.0 / E_)
    zero_f64 = ct.full((1, 1), 0.0, dtype=ct.float64)
    variance = ct.astype(ct.where(var64 < 0.0, zero_f64, var64), ct.float32)
    mean = ct.astype(mean64, ct.float32)
    invstd = ct.rsqrt(variance + EPS)

    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(channel,), tile=ct.reshape(invstd, (1,)))


@ct.kernel
def _bn_affine_relu_kernel(
    x_ptr,             # bf16 [num_tiles, BLOCK_E] (padded flat NCHW)
    mean_bc_ptr,       # f32 same layout
    invstd_bc_ptr,     # f32 same layout
    weight_bc_ptr,     # f32 same layout
    bias_bc_ptr,       # f32 same layout
    out_ptr,           # bf16 same layout
    BLOCK_E_: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_E_))
    mean = ct.load(mean_bc_ptr, index=(row, 0), shape=(1, BLOCK_E_))
    invstd = ct.load(invstd_bc_ptr, index=(row, 0), shape=(1, BLOCK_E_))
    weight = ct.load(weight_bc_ptr, index=(row, 0), shape=(1, BLOCK_E_))
    bias = ct.load(bias_bc_ptr, index=(row, 0), shape=(1, BLOCK_E_))

    x64 = ct.astype(x, ct.float64)
    mean64 = ct.astype(mean, ct.float64)
    invstd64 = ct.astype(invstd, ct.float64)
    weight64 = ct.astype(weight, ct.float64)
    bias64 = ct.astype(bias, ct.float64)

    affine = ((x64 - mean64) * invstd64) * weight64 + bias64
    rounded_f32 = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    is_positive = rounded_f32 > 0.0
    is_nan = rounded_f32 != rounded_f32
    keep = is_positive | is_nan
    relu = ct.where(keep, rounded_f32, 0.0)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(relu, ct.bfloat16))


@oracle_impl(hardware="B200", point="b0183d97", BLOCK_K=8192, BLOCK_E=1024)
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape0, _shape1 = inputs
    device = arg0_1.device

    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw_size = height * width
    elements_per_channel = batch * hw_size
    total_elements = batch * channels * hw_size
    out_channels = channels * 2

    # 1. Gather per-channel tiles: [C, elements_per_channel] contiguous.
    x_gathered = arg0_1.permute(1, 0, 2, 3).contiguous().reshape(
        channels, elements_per_channel
    )

    mean = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, channels, 1, 1),
        (channels, 1, 1, 1),
        device=device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, 1, 1), _bn_stats_from_gathered_kernel,
        (x_gathered, mean.view(channels), invstd.view(channels),
         elements_per_channel, BLOCK_K),
    )

    # Running-stats update (torch, matches Triton's fp32 formula).
    mean_1d = mean.view(channels)
    invstd_1d = invstd.view(channels)
    variance_1d = 1.0 / (invstd_1d * invstd_1d) - EPS

    new_running_mean = mean_1d * MOMENTUM + arg1_1 * (1.0 - MOMENTUM)
    new_running_var = (variance_1d * RUNNING_VAR_CORRECTION) * MOMENTUM + arg2_1 * (1.0 - MOMENTUM)
    torch.ops.aten.copy_(arg1_1, new_running_mean)
    torch.ops.aten.copy_(arg2_1, new_running_var)

    # 2. BN affine + ReLU on contiguous NCHW x -> flat bf16.
    weight_1d = arg3_1
    bias_1d = arg4_1

    def broadcast_channel(v):
        return v.view(1, channels, 1, 1).expand(batch, channels, height, width).contiguous().view(-1)

    x_flat = arg0_1.contiguous().view(-1)
    mean_bc = broadcast_channel(mean_1d)
    invstd_bc = broadcast_channel(invstd_1d)
    weight_bc = broadcast_channel(weight_1d)
    bias_bc = broadcast_channel(bias_1d)

    num_tiles = (total_elements + BLOCK_E - 1) // BLOCK_E
    padded = num_tiles * BLOCK_E

    def pad_bf16(t):
        buf = torch.zeros(padded, device=device, dtype=torch.bfloat16)
        buf[:total_elements] = t
        return buf.view(num_tiles, BLOCK_E)

    def pad_f32(t):
        buf = torch.zeros(padded, device=device, dtype=torch.float32)
        buf[:total_elements] = t
        return buf.view(num_tiles, BLOCK_E)

    x_pad = pad_bf16(x_flat)
    mean_pad = pad_f32(mean_bc)
    invstd_pad = pad_f32(invstd_bc)
    weight_pad = pad_f32(weight_bc)
    bias_pad = pad_f32(bias_bc)
    relu_pad = torch.empty(num_tiles, BLOCK_E, device=device, dtype=torch.bfloat16)

    ct.launch(
        stream, (num_tiles, 1, 1), _bn_affine_relu_kernel,
        (x_pad, mean_pad, invstd_pad, weight_pad, bias_pad, relu_pad, BLOCK_E),
    )

    relu_out = relu_pad.view(-1)[:total_elements].view(batch, channels, height, width)
    skip_contig = arg5_1.contiguous()

    # Build the interleaved [N, 2C, H, W] output: for each c, output channel 2c
    # holds skip[c] and 2c+1 holds relu[c] (per Triton's out_base formula).
    shuffled = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * hw_size, hw_size, width, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    shuffled_view = shuffled.view(batch, channels, 2, height, width)
    shuffled_view[:, :, 0, :, :].copy_(skip_contig)
    shuffled_view[:, :, 1, :, :].copy_(relu_out)

    return (
        mean,
        invstd,
        shuffled[:, :channels, :, :],
        shuffled[:, channels:, :, :],
        arg1_1,
        arg2_1,
    )
