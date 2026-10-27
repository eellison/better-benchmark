"""cuTile port of var_mean_mean_3f4aa90365fa: MobileNetV3 BN training + hardswish.

Two-kernel: (a) per-channel var/mean over all NHW positions and running stat
update, (b) hardswish + spatial mean per (n, c).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_stats_kernel(
    x_ptr,             # bf16 [N, C, H, W] (contiguous)
    running_mean_ptr,  # f32 [C]
    running_var_ptr,   # f32 [C]
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    N: ct.Constant[int],
    HW: ct.Constant[int],
    E: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    channel = ct.bid(0)
    # Load tile of [BLOCK_N, 1, BLOCK_HW] with x[b, channel, hw].
    # Instead use a 3D view: shape [N, C, HW]. We tile-load [1, 1, BLOCK_HW]
    # per program launched over (channel, tile_n). Sum all.
    # We can just tile-load [BLOCK_N, 1, BLOCK_HW] if BLOCK_N=N & BLOCK_HW=HW.
    tile = ct.load(x_ptr, index=(0, channel, 0), shape=(BLOCK_N, 1, BLOCK_HW),
                   padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(tile, ct.float32)
    sum_x = ct.sum(x_f)
    sum_x2 = ct.sum(x_f * x_f)
    inv_e = 1.0 / E
    mean = sum_x * inv_e
    var = sum_x2 * inv_e - mean * mean
    zero = ct.full(shape=(1,), fill_value=0.0, dtype=ct.float32)
    var = ct.where(var > 0.0, var, zero)
    invstd = ct.rsqrt(var + 0.001)

    old_mean = ct.load(running_mean_ptr, index=(channel,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(channel,), shape=(1,))
    updated_mean = mean * 0.01 + old_mean * 0.99
    corrected_var = var * 1.0006381620931717
    updated_var = corrected_var * 0.01 + old_var * 0.99

    ct.store(mean_ptr, index=(channel,), tile=mean)
    ct.store(invstd_ptr, index=(channel,), tile=invstd)
    ct.store(running_mean_ptr, index=(channel,), tile=updated_mean)
    ct.store(running_var_ptr, index=(channel,), tile=updated_var)


@ct.kernel
def _hardswish_spatial_mean_kernel(
    x_ptr,            # bf16 [N, C, HW]
    mean_ptr,         # f32 [C]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    bias_ptr,         # f32 [C]
    activation_ptr,   # bf16 [N, C, HW]
    spatial_mean_ptr, # bf16 [N, C]
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    m3 = ct.reshape(mean, (1, 1, 1))
    i3 = ct.reshape(invstd, (1, 1, 1))
    w3 = ct.reshape(weight, (1, 1, 1))
    b3 = ct.reshape(bias, (1, 1, 1))
    normalized = (x_f - m3) * i3
    affine = normalized * w3 + b3
    rounded_bf = ct.astype(affine, ct.bfloat16)
    rounded = ct.astype(rounded_bf, ct.float32)
    shifted = rounded + 3.0
    zero_val = ct.full(shape=(1, 1, 1), fill_value=0.0, dtype=ct.float32)
    six_val = ct.full(shape=(1, 1, 1), fill_value=6.0, dtype=ct.float32)
    clamp_low = ct.where(shifted > 0.0, shifted, zero_val)
    clamp_up = ct.where(clamp_low < 6.0, clamp_low, six_val)
    hardswish = rounded * clamp_up / 6.0
    hardswish_bf16 = ct.astype(hardswish, ct.bfloat16)
    ct.store(activation_ptr, index=(n, c, 0), tile=hardswish_bf16)

    # Spatial mean: sum bf16-rounded hardswish values over HW dim, divide by HW.
    hw_f = ct.astype(hardswish_bf16, ct.float32)
    ks = ct.arange(BLOCK_HW, dtype=ct.int32)
    in_bounds = ks < HW
    zero_2d = ct.full(shape=(1, 1, BLOCK_HW), fill_value=0.0, dtype=ct.float32)
    hw_masked = ct.where(ct.reshape(in_bounds, (1, 1, BLOCK_HW)), hw_f, zero_2d)
    total = ct.sum(hw_masked)
    mean_val = ct.astype(total * (1.0 / HW), ct.bfloat16)
    ct.store(spatial_mean_ptr, index=(n, c), tile=mean_val)


@oracle_impl(hardware="B200", point="e8a9d13a")
@oracle_impl(hardware="B200", point="d3bf7421")
@oracle_impl(hardware="B200", point="8aafc432")
@oracle_impl(hardware="B200", point="ba044ce1")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw

    saved_mean = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.float32,
    )
    activation = torch.empty(
        (n, c, h, w), device=x.device, dtype=torch.bfloat16,
    )
    spatial_mean = torch.empty_strided(
        (n, c, 1, 1), (c, 1, 1, 1), device=x.device, dtype=torch.bfloat16,
    )

    # Views for 3D operation.
    x_3d = x.view(n, c, hw)
    activation_3d = activation.view(n, c, hw)
    saved_mean_1d = saved_mean.view(c)
    invstd_1d = invstd.view(c)
    spatial_mean_2d = spatial_mean.view(n, c)

    # Choose BLOCK sizes: BLOCK_N = pow2 >= n, BLOCK_HW = pow2 >= hw.
    block_n = 1
    while block_n < n:
        block_n *= 2
    block_hw = 1
    while block_hw < hw:
        block_hw *= 2

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1),
        _bn_stats_kernel,
        (x_3d, running_mean, running_var, saved_mean_1d, invstd_1d,
         n, hw, e, block_n, block_hw),
    )
    ct.launch(
        stream, (n, c, 1),
        _hardswish_spatial_mean_kernel,
        (x_3d, saved_mean_1d, invstd_1d, weight, bias, activation_3d,
         spatial_mean_2d, hw, block_hw),
    )
    return saved_mean, invstd, activation, spatial_mean, running_mean, running_var
