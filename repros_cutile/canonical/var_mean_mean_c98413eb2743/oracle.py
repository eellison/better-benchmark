"""cuTile port of var_mean_mean_c98413eb2743: MobileNetV3 BN-training + running-stat copy_.

The heavy per-channel reduction (mean/var over NHW), affine + ReLU, and the
spatial-mean epilogue run through cuTile. The `copy_` in-place mutations of
running_mean/running_var use torch.ops.aten.copy_ from oracle_forward with
values computed on the host side.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001
MOMENTUM = 0.01


@ct.kernel
def _reduce_channel_kernel(
    x_ptr,             # bf16 [N, C, HW]  (contiguous)
    sum_ptr,           # f32  [C]
    sumsq_ptr,         # f32  [C]
    N: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_PADDED: ct.Constant[int],
):
    channel = ct.bid(0)

    acc_sum = ct.zeros((1,), dtype=ct.float32)
    acc_sumsq = ct.zeros((1,), dtype=ct.float32)

    for n in range(N):
        x = ct.load(
            x_ptr, index=(n, channel, 0), shape=(1, 1, HW_PADDED),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_1d = ct.reshape(x, (HW_PADDED,))
        x_f = ct.astype(x_1d, ct.float32)
        # HW_PADDED >= HW, and padding-mode zero zeros the tail, so no explicit mask.
        acc_sum = acc_sum + ct.reshape(ct.sum(x_f), (1,))
        acc_sumsq = acc_sumsq + ct.reshape(ct.sum(x_f * x_f), (1,))

    ct.store(sum_ptr, index=(channel,), tile=acc_sum)
    ct.store(sumsq_ptr, index=(channel,), tile=acc_sumsq)


@ct.kernel
def _bn_relu_kernel(
    x_ptr,             # bf16 [N, C, HW]
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    bias_ptr,          # f32 [C]
    relu_ptr,          # bf16 [N, C, HW]
    spatial_ptr,       # bf16 [N, C]
    HW: ct.Constant[int],
    HW_PADDED: ct.Constant[int],
    INV_HW: ct.Constant[float],
):
    n = ct.bid(0)
    channel = ct.bid(1)

    x = ct.load(
        x_ptr, index=(n, channel, 0), shape=(1, 1, HW_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(ct.reshape(x, (HW_PADDED,)), ct.float32)

    mean = ct.load(mean_ptr, index=(channel,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(channel,), shape=(1,))
    weight = ct.load(weight_ptr, index=(channel,), shape=(1,))
    bias = ct.load(bias_ptr, index=(channel,), shape=(1,))
    mean_b = ct.broadcast_to(mean, (HW_PADDED,))
    invstd_b = ct.broadcast_to(invstd, (HW_PADDED,))
    weight_b = ct.broadcast_to(weight, (HW_PADDED,))
    bias_b = ct.broadcast_to(bias, (HW_PADDED,))

    y = (x_f - mean_b) * invstd_b * weight_b + bias_b
    y_bf = ct.astype(y, ct.bfloat16)
    zero_bf = ct.full((HW_PADDED,), 0.0, dtype=ct.bfloat16)
    # NaN-preserving ReLU (aten.relu: NaN in -> NaN out).
    is_nan = ct.astype(y_bf, ct.float32) != ct.astype(y_bf, ct.float32)
    relu_bf = ct.where(is_nan, y_bf, ct.where(y_bf > zero_bf, y_bf, zero_bf))

    # Masked scatter of valid HW positions.
    cols = ct.arange(HW_PADDED, dtype=ct.int32)
    hw_valid = cols < HW
    base = (n * ct.num_blocks(1) + channel) * HW
    idx = base + cols
    ct.scatter(relu_ptr, (idx,), relu_bf, mask=hw_valid)

    # Spatial mean: sum valid, / HW, cast to bf16.
    zero_f = ct.full((HW_PADDED,), 0.0, dtype=ct.float32)
    relu_f = ct.astype(relu_bf, ct.float32)
    relu_masked = ct.where(hw_valid, relu_f, zero_f)
    spatial_sum = ct.sum(relu_masked)
    spatial_mean = ct.astype(
        ct.reshape(spatial_sum * INV_HW, (1,)), ct.bfloat16
    )
    spatial_off = n * ct.num_blocks(1) + channel
    ct.store(spatial_ptr, index=(spatial_off,), tile=spatial_mean)


def _next_pow2(x):
    v = 1
    while v < x:
        v <<= 1
    return v


def _launch(inputs):
    x, running_mean, running_var, weight, bias = inputs
    device = x.device
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w
    e = n * hw
    hw_padded = _next_pow2(hw)

    x_contig = x.contiguous()
    x_flat = x_contig.view(n, c, hw)

    sum_c = torch.zeros(c, device=device, dtype=torch.float32)
    sumsq_c = torch.zeros(c, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _reduce_channel_kernel,
        (x_flat, sum_c, sumsq_c, n, c, hw, hw_padded),
    )

    mean_c = sum_c / e
    var_c = sumsq_c / e - mean_c * mean_c
    var_c = torch.clamp(var_c, min=0.0)
    invstd_c = torch.rsqrt(var_c + EPS)

    # copy_ running stats.
    correction = e / (e - 1.0)
    new_running_mean = running_mean * 0.99 + mean_c * 0.01
    new_running_var = running_var * 0.99 + (var_c * correction) * 0.01
    torch.ops.aten.copy_.default(running_mean, new_running_mean)
    torch.ops.aten.copy_.default(running_var, new_running_var)

    saved_mean = mean_c.view(1, c, 1, 1)
    invstd_out = invstd_c.view(1, c, 1, 1)

    relu = torch.empty((n, c, h, w), device=device, dtype=torch.bfloat16)
    spatial = torch.empty((n, c), device=device, dtype=torch.bfloat16)

    ct.launch(
        stream,
        (n, c, 1),
        _bn_relu_kernel,
        (
            x_flat, mean_c, invstd_c, weight, bias,
            relu.view(-1), spatial.view(-1),
            hw, hw_padded, 1.0 / hw,
        ),
    )

    spatial_mean = spatial.view(n, c, 1, 1)

    return saved_mean, invstd_out, relu, spatial_mean, running_mean, running_var


@oracle_impl(hardware="B200", point="00cca0ad")
@oracle_impl(hardware="B200", point="b7fea7d6")
def oracle_forward(inputs):
    return _launch(inputs)
