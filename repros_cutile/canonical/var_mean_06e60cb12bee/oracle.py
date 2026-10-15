"""cuTile port of var_mean_06e60cb12bee: UNet training-BN + bilinear-upsample-cat.

Uses cuTile for the per-channel var_mean reduction, then computes the rest
(BN affine + ReLU + bilinear upsample + pad + cat + running-stat mutation)
in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
VAR_CORRECTION = 1.0004239084357778
MOMENTUM_NEW = 0.1
MOMENTUM_OLD = 0.9


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _channel_var_mean_kernel(
    x_ptr,          # bf16 [C, N_ELEMS] (viewed contiguously)
    mean_ptr,       # f32  [C]
    var_ptr,        # f32  [C]
    N_ELEMS: ct.Constant[int],
    N_ELEMS_F: ct.Constant[float],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    x = ct.load(
        x_ptr, index=(channel, 0), shape=(1, BLOCK_R),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    cols = ct.arange(BLOCK_R, dtype=ct.int32)
    valid = cols < N_ELEMS
    valid_2d = ct.reshape(valid, (1, BLOCK_R))
    zero_f = ct.full((1, BLOCK_R), 0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x_f, zero_f)
    total = ct.sum(x_masked)
    total_sq = ct.sum(x_masked * x_masked)
    mean = total * (1.0 / N_ELEMS_F)
    var = total_sq * (1.0 / N_ELEMS_F) - mean * mean
    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(var_ptr, index=(channel,), tile=ct.reshape(var, (1,)))


@oracle_impl(hardware="B200", point="7ff155f1")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0 = inputs
    device = arg0_1.device
    channels = int(arg0_1.shape[1])
    in_h = int(arg0_1.shape[2])
    in_w = int(arg0_1.shape[3])
    n_elems = in_h * in_w
    block_r = _next_pow2(n_elems)

    mean_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    var_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    x_2d = arg0_1.reshape(channels, n_elems)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, 1, 1), _channel_var_mean_kernel,
        (x_2d, mean_1d, var_1d, n_elems, float(n_elems), block_r),
    )
    mean = mean_1d.view(1, channels, 1, 1)
    var = var_1d.view(1, channels, 1, 1)
    rsqrt = torch.rsqrt(var + EPS)

    normalized = (arg0_1 - mean) * rsqrt
    weight_ = arg3_1.view(1, channels, 1, 1)
    bias_ = arg4_1.view(1, channels, 1, 1)
    affine = normalized * weight_ + bias_
    affine_bf16 = affine.to(torch.bfloat16)
    relu = torch.relu(affine_bf16)
    relu_f32 = relu.to(torch.float32)

    iota_h = torch.arange(80, device=device, dtype=torch.int64)
    row_f = iota_h.to(torch.float32) * 0.4936708860759494
    row_f = torch.clamp_min(row_f, 0.0).view(80, 1)
    row0 = row_f.to(torch.int64)
    row1 = torch.clamp_max(row0 + 1, 39)
    row_delta = torch.clamp_max(torch.clamp_min(row_f - row0.to(torch.float32), 0.0), 1.0)

    iota_w = torch.arange(118, device=device, dtype=torch.int64)
    col_f = iota_w.to(torch.float32) * 0.49572649572649574
    col_f = torch.clamp_min(col_f, 0.0)
    col0 = col_f.to(torch.int64)
    col1 = torch.clamp_max(col0 + 1, 58)
    col_delta = torch.clamp_max(torch.clamp_min(col_f - col0.to(torch.float32), 0.0), 1.0)

    v00 = relu_f32[:, :, row0[:, 0], :][:, :, :, col0]
    v01 = relu_f32[:, :, row0[:, 0], :][:, :, :, col1]
    v10 = relu_f32[:, :, row1[:, 0], :][:, :, :, col0]
    v11 = relu_f32[:, :, row1[:, 0], :][:, :, :, col1]

    top = v00 + (v01 - v00) * col_delta
    bottom = v10 + (v11 - v10) * col_delta
    interp = top + (bottom - top) * row_delta
    interp_bf = interp.to(torch.bfloat16)
    padded = torch.nn.functional.pad(interp_bf, (0, 1, 0, 0), value=0.0)
    cat = torch.cat([arg5_1, padded], dim=1)

    new_running_mean = mean_1d * MOMENTUM_NEW + arg1_1 * MOMENTUM_OLD
    new_running_var = (var_1d * VAR_CORRECTION) * MOMENTUM_NEW + arg2_1 * MOMENTUM_OLD
    arg1_1.copy_(new_running_mean)
    arg2_1.copy_(new_running_var)

    return (
        mean, rsqrt, row0, row1, col0, col1,
        col_delta, row_delta, cat, arg1_1, arg2_1,
    )
