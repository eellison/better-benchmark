"""cuTile port of var_mean_mean_1dc49728bc38: DenseNet BN training + ReLU + spatial mean.

Per-channel population var_mean over [N, H, W], BN affine, ReLU, then per-image
per-channel spatial mean. Running mean/var updates use torch.copy_ outside of
the cuTile kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _bn_relu_spatial_mean_kernel(
    x_ptr,           # bf16 [N, C, HW]  (contiguous view)
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    mean_out_ptr,    # f32 [C]
    invstd_out_ptr,  # f32 [C]
    spatial_mean_ptr,# bf16 [N, C]
    HW_PAD: ct.Constant[int],
    HW_ACTUAL: ct.Constant[int],
    N_BLOCK: ct.Constant[int],
    INV_E: ct.Constant[float],
    INV_HW: ct.Constant[float],
):
    channel = ct.bid(0)
    x = ct.load(
        x_ptr,
        index=(0, channel, 0),
        shape=(N_BLOCK, 1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    x_f = ct.reshape(x_f, (N_BLOCK, HW_PAD))
    cols = ct.arange(HW_PAD, dtype=ct.int32)
    col_mask = cols < HW_ACTUAL
    col_mask_2d = ct.reshape(col_mask, (1, HW_PAD))
    x_masked = ct.where(col_mask_2d, x_f, 0.0)

    total = ct.sum(x_masked)
    mean = total * INV_E
    centered = ct.where(col_mask_2d, x_f - mean, 0.0)
    var = ct.sum(centered * centered) * INV_E
    invstd = ct.rsqrt(var + EPS)

    weight_tile = ct.load(weight_ptr, index=(channel,), shape=(1,))
    bias_tile = ct.load(bias_ptr, index=(channel,), shape=(1,))
    gamma = ct.reshape(weight_tile, ())
    beta = ct.reshape(bias_tile, ())

    normalized = centered * invstd
    scaled = normalized * gamma + beta
    scaled_bf = ct.astype(scaled, ct.bfloat16)
    scaled_bf_f = ct.astype(scaled_bf, ct.float32)
    relu_val = ct.where(scaled_bf_f > 0.0, scaled_bf_f, 0.0)
    relu_masked = ct.where(col_mask_2d, relu_val, 0.0)
    pooled = ct.sum(relu_masked, axis=1) * INV_HW
    pooled_bf = ct.astype(pooled, ct.bfloat16)

    ct.store(mean_out_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_out_ptr, index=(channel,), tile=ct.reshape(invstd, (1,)))
    pooled_2d = ct.reshape(pooled_bf, (N_BLOCK, 1))
    ct.store(spatial_mean_ptr, index=(0, channel), tile=pooled_2d)


@oracle_impl(hardware="B200", point="1a932376")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, _shape_param_0 = inputs
    n, c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    hw = h * w
    e = n * hw
    pad_hw = 1 << (hw - 1).bit_length() if hw & (hw - 1) else hw

    device = x.device
    mean = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32)
    invstd = torch.empty_strided((1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32)
    spatial_mean = torch.empty_strided((n, c), (c, 1), device=device, dtype=torch.bfloat16)

    x_flat = x.view(n, c, hw)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_relu_spatial_mean_kernel,
        (
            x_flat,
            weight,
            bias,
            mean.view(-1),
            invstd.view(-1),
            spatial_mean,
            pad_hw,
            hw,
            n,
            1.0 / e,
            1.0 / hw,
        ),
    )

    running_var_correction = e / (e - 1.0)
    mean_1d = mean.view(-1)
    var_1d = 1.0 / (invstd.view(-1) ** 2) - EPS
    new_mean = running_mean * 0.9 + mean_1d * 0.1
    new_var = running_var * 0.9 + (var_1d * running_var_correction) * 0.1
    running_mean.copy_(new_mean)
    running_var.copy_(new_var)
    return mean, invstd, spatial_mean, running_mean, running_var
