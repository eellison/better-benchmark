"""cuTile port of var_mean_mean_19ffdb35cc67: ShuffleNet BN + ReLU + spatial mean.

Two cuTile kernels:
  1. per-channel BN statistics (mean/var/invstd) over N*H*W elements.
  2. per (N, C) row: BN affine ReLU (bf16 rounding boundary) + spatial mean.

torch handles the running_mean/running_var copy_ update after kernel 1.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@ct.kernel
def _bn_stats_kernel(
    x_ptr,           # bf16 [C, N*H*W] channel-major
    saved_mean_ptr,  # f32 [C]
    invstd_ptr,      # f32 [C]
    var_ptr,         # f32 [C]   population variance
    ELEMS: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
    EPS_: ct.Constant[float],
):
    c = ct.bid(0)
    row = ct.load(x_ptr, index=(c, 0), shape=(1, BLOCK_E),
                  padding_mode=ct.PaddingMode.ZERO)
    row_f = ct.astype(row, ct.float32)

    cols = ct.arange(BLOCK_E, dtype=ct.int32)
    valid = ct.reshape(cols < ELEMS, (1, BLOCK_E))
    zero_f = ct.zeros((1, BLOCK_E), dtype=ct.float32)
    row_masked = ct.where(valid, row_f, zero_f)

    inv_n = 1.0 / ELEMS
    sum_x = ct.sum(row_masked, axis=1, keepdims=True)
    mean = sum_x * inv_n
    sum_x2 = ct.sum(row_masked * row_masked, axis=1, keepdims=True)
    variance = sum_x2 * inv_n - mean * mean
    variance = ct.where(variance > 0.0, variance, 0.0)
    invstd = ct.rsqrt(variance + EPS_)

    ct.store(saved_mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))
    ct.store(var_ptr, index=(c,), tile=ct.reshape(variance, (1,)))


@ct.kernel
def _bn_affine_relu_spatial_mean_kernel(
    x_ptr,             # bf16 [N*C, H*W] flat (row-major)
    saved_mean_ptr,    # f32 [C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    bias_ptr,          # f32 [C]
    spatial_mean_ptr,  # bf16 [N*C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)
    # channel = row % C
    row_idx_full = ct.full((1,), row, dtype=ct.int32)
    c_idx = row_idx_full - (row_idx_full // C_) * C_

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    cols = ct.arange(BLOCK_HW, dtype=ct.int32)
    valid = ct.reshape(cols < HW_, (1, BLOCK_HW))
    zero_f = ct.zeros((1, BLOCK_HW), dtype=ct.float32)

    mean = ct.gather(saved_mean_ptr, c_idx)
    invstd = ct.gather(invstd_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    bias = ct.gather(bias_ptr, c_idx)
    mean_2d = ct.reshape(mean, (1, 1))
    invstd_2d = ct.reshape(invstd, (1, 1))
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))

    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((1, BLOCK_HW), dtype=ct.bfloat16)
    relu_bf = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)
    relu_f = ct.astype(relu_bf, ct.float32)
    relu_masked = ct.where(valid, relu_f, zero_f)

    inv_hw = 1.0 / HW_
    total = ct.sum(relu_masked, axis=1, keepdims=True)
    spatial_mean_bf = ct.astype(total * inv_hw, ct.bfloat16)
    ct.store(spatial_mean_ptr, index=(row,),
             tile=ct.reshape(spatial_mean_bf, (1,)))


@oracle_impl(hardware="B200", point="5b9efb9c")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias = inputs
    device = x.device
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    elements_per_channel = n * hw

    # Channel-major view: (C, N*H*W).
    x_channel_major = (
        x.permute(1, 0, 2, 3).contiguous().view(c, elements_per_channel)
    )

    saved_mean_1d = torch.empty((c,), device=device, dtype=torch.float32)
    invstd_1d = torch.empty((c,), device=device, dtype=torch.float32)
    variance_1d = torch.empty((c,), device=device, dtype=torch.float32)

    block_e = _next_pow2(elements_per_channel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1), _bn_stats_kernel,
        (x_channel_major, saved_mean_1d, invstd_1d, variance_1d,
         elements_per_channel, block_e, EPS),
    )

    # BN affine + ReLU + spatial mean kernel.
    x_flat_rows = x.contiguous().view(n * c, hw)
    spatial_mean_flat = torch.empty((n * c,), device=device, dtype=torch.bfloat16)
    block_hw = _next_pow2(hw)
    ct.launch(
        stream, (n * c, 1, 1), _bn_affine_relu_spatial_mean_kernel,
        (x_flat_rows, saved_mean_1d, invstd_1d, weight, bias,
         spatial_mean_flat, c, hw, block_hw),
    )
    spatial_mean = spatial_mean_flat.view(n, c)

    # Running stats update: population variance correction factor
    # matches the eager repro's 1.0001594642002871 for N*H*W=6272.
    if elements_per_channel > 1:
        var_correction = elements_per_channel / (elements_per_channel - 1)
    else:
        var_correction = 1.0
    new_mean = running_mean * (1.0 - MOMENTUM) + saved_mean_1d * MOMENTUM
    new_var = running_var * (1.0 - MOMENTUM) + variance_1d * var_correction * MOMENTUM
    torch.ops.aten.copy_(running_mean, new_mean)
    torch.ops.aten.copy_(running_var, new_var)

    # Returned f32 tensors are shaped (1, C, 1, 1).
    saved_mean_out = saved_mean_1d.view(1, c, 1, 1).contiguous()
    invstd_out = invstd_1d.view(1, c, 1, 1).contiguous()
    return saved_mean_out, invstd_out, spatial_mean, running_mean, running_var
