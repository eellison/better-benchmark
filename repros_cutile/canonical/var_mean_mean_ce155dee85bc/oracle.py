"""cuTile port of var_mean_mean_ce155dee85bc: GhostNet BN training + spatial mean.

Approach:
  1. `_bn_stats_kernel`: per-channel fp32 population var_mean over N*H*W (Welford
     via sum-of-squares to match Triton), stores saved_mean and invstd.
     Gather x into a `[C, N*H*W]` layout so each channel is one row.
  2. `_affine_spatial_mean_kernel`: per (n, c) row of H*W elements — BN affine
     (fp32), bf16 cast, spatial mean.
  3. torch epilogue: running-stat update; broadcast the affine output into
     channels-last strides.

Input strides are channels-last (bf16), so we permute to NHWC before gather.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361


@ct.kernel
def _bn_stats_from_gathered_kernel(
    xg_ptr,          # bf16 [C, E]
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
    x_f32 = ct.astype(xg, ct.float32)
    col_idx = ct.arange(BLOCK_K_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < E_, (1, BLOCK_K_))
    x_masked = ct.where(col_mask, x_f32, 0.0)
    total = ct.sum(x_masked)
    total2 = ct.sum(x_masked * x_masked)
    mean = total * (1.0 / E_)
    var = total2 * (1.0 / E_) - mean * mean
    zero_f32 = ct.full((1, 1), 0.0, dtype=ct.float32)
    variance = ct.where(var < 0.0, zero_f32, var)
    invstd = ct.rsqrt(variance + EPS)
    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(channel,), tile=ct.reshape(invstd, (1,)))


@ct.kernel
def _affine_spatial_mean_kernel(
    xg_ptr,           # bf16 [total_rows=N*C, HW]  (gathered by (n, c) then flat h*w)
    mean_ptr,         # f32 [total_rows]           (broadcast per row = channel)
    invstd_ptr,       # f32 [total_rows]
    weight_ptr,       # f32 [total_rows]
    bias_ptr,         # f32 [total_rows]
    out_ptr,          # bf16 [total_rows, HW]
    spatial_mean_ptr, # bf16 [total_rows]
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        xg_ptr, index=(row, 0), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean = ct.load(mean_ptr, index=(row,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))
    weight = ct.load(weight_ptr, index=(row,), shape=(1,))
    bias = ct.load(bias_ptr, index=(row,), shape=(1,))

    mean_2d = ct.reshape(mean, (1, 1))
    invstd_2d = ct.reshape(invstd, (1, 1))
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))

    x_f32 = ct.astype(x, ct.float32)
    normalized = (x_f32 - mean_2d) * invstd_2d
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=affine_bf16)

    # Spatial mean over HW: sum then divide by HW.
    col_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HW_, (1, BLOCK_HW_))
    x_hw = ct.astype(affine_bf16, ct.float32)
    x_masked = ct.where(col_mask, x_hw, 0.0)
    total = ct.sum(x_masked)
    mean_value = ct.astype(total * (1.0 / HW_), ct.bfloat16)
    ct.store(spatial_mean_ptr, index=(row,), tile=ct.reshape(mean_value, (1,)))


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="c5a5573b", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=64, ROW_BLOCK=32)
@oracle_impl(hardware="B200", point="0cb3f08a", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=1024, ROW_BLOCK=8)
def oracle_forward(inputs, **_kwargs):
    x, running_mean, running_var, weight, bias = inputs
    device = x.device
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw_size = h * w
    elements_per_channel = n * hw_size
    total_rows = n * c

    # Gather to (C, N*H*W) contiguous for stats.
    x_gathered = x.permute(1, 0, 2, 3).contiguous().reshape(c, elements_per_channel)

    saved_mean = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    # Round up BLOCK_K to a power of 2 ≥ elements_per_channel.
    block_k = _next_power_of_2(max(elements_per_channel, 1))
    ct.launch(
        stream, (c, 1, 1), _bn_stats_from_gathered_kernel,
        (x_gathered, saved_mean.view(c), invstd.view(c), elements_per_channel, block_k),
    )

    # Running stats update via torch (matches Triton fp32 arithmetic).
    mean_1d = saved_mean.view(c)
    invstd_1d = invstd.view(c)
    variance_1d = 1.0 / (invstd_1d * invstd_1d) - EPS
    variance_1d = torch.clamp(variance_1d, min=0.0)
    new_running_mean = mean_1d * MOMENTUM + running_mean * (1.0 - MOMENTUM)
    corrected_var = variance_1d * RUNNING_VAR_CORRECTION
    new_running_var = corrected_var * MOMENTUM + running_var * (1.0 - MOMENTUM)
    torch.ops.aten.copy_(running_mean, new_running_mean)
    torch.ops.aten.copy_(running_var, new_running_var)

    # 2. Affine + spatial mean: gather per (n, c) → (N*C, HW) contiguous.
    x_gathered_nc = x.permute(0, 1, 2, 3).contiguous().view(n, c, hw_size).view(total_rows, hw_size)
    # Note: x.permute(0, 1, 2, 3) is a no-op for shape but forces contig fetch.

    # Broadcast per-row (per-channel) scalars into [N*C] arrays.
    mean_bc = mean_1d.view(1, c, 1).expand(n, c, 1).contiguous().view(-1)
    invstd_bc = invstd_1d.view(1, c, 1).expand(n, c, 1).contiguous().view(-1)
    weight_bc = weight.view(1, c, 1).expand(n, c, 1).contiguous().view(-1)
    bias_bc = bias.view(1, c, 1).expand(n, c, 1).contiguous().view(-1)

    # Use a rounded-up BLOCK_HW = power of 2 ≥ hw_size (constants must be power-of-2 for tiles).
    block_hw = _next_power_of_2(max(hw_size, 1))

    out_gathered = torch.empty((total_rows, hw_size), device=device, dtype=torch.bfloat16)
    spatial_mean_1d = torch.empty(total_rows, device=device, dtype=torch.bfloat16)

    ct.launch(
        stream, (total_rows, 1, 1), _affine_spatial_mean_kernel,
        (x_gathered_nc, mean_bc, invstd_bc, weight_bc, bias_bc,
         out_gathered, spatial_mean_1d, hw_size, block_hw),
    )

    # Restore channels-last stride for activation.
    activation = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=device,
        dtype=torch.bfloat16,
    )
    activation.copy_(out_gathered.view(n, c, h, w))

    spatial_mean = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    spatial_mean.view(-1).copy_(spatial_mean_1d)

    return saved_mean, invstd, activation, spatial_mean, running_mean, running_var
