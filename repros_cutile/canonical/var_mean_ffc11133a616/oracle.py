"""cuTile port of var_mean_ffc11133a616: NFNet weight-standardization LayerNorm-like scaling.

Per-channel row layer-norm over `inner_size` values, then `gain * scale` bf16-rounded factor.
Uses zero-padded loads + masked sums to handle non-power-of-2 inner sizes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
SCALE = 0.02551551815399144


@ct.kernel
def _channel_norm_scale_bf16_kernel(
    weight_ptr,
    gain_ptr,
    out_ptr,
    INNER_SIZE: ct.Constant[int],
    INNER_PAD: ct.Constant[int],
):
    channel = ct.bid(0)

    x_bf = ct.load(
        weight_ptr,
        index=(channel, 0),
        shape=(1, INNER_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)

    # Build mask to zero-out padding beyond INNER_SIZE.
    cols = ct.arange(INNER_PAD, dtype=ct.int32)
    valid = ct.reshape(cols < INNER_SIZE, (1, INNER_PAD))
    x_for_sum = ct.where(
        valid,
        x,
        ct.full(shape=(1, INNER_PAD), fill_value=0.0, dtype=ct.float32),
    )

    mean = ct.sum(x_for_sum) / INNER_SIZE
    centered = x - mean  # values outside INNER_SIZE will be -mean, ignore below.
    centered_masked = ct.where(
        valid,
        centered,
        ct.full(shape=(1, INNER_PAD), fill_value=0.0, dtype=ct.float32),
    )
    variance = ct.sum(centered_masked * centered_masked) / INNER_SIZE
    invstd = ct.rsqrt(variance + EPS)

    gain = ct.astype(ct.load(gain_ptr, index=(channel,), shape=(1,)), ct.float32)
    gain_scaled_bf = ct.astype(gain * SCALE, ct.bfloat16)
    gain_scaled = ct.astype(gain_scaled_bf, ct.float32)

    normalized = centered * invstd
    out = normalized * gain_scaled
    ct.store(out_ptr, index=(channel, 0), tile=ct.astype(out, ct.bfloat16))


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="a24dc267")
@oracle_impl(hardware="B200", point="b321bd29")
@oracle_impl(hardware="B200", point="e5403093")
@oracle_impl(hardware="B200", point="54d002a2")
@oracle_impl(hardware="B200", point="1c377997")
@oracle_impl(hardware="B200", point="067f0896")
@oracle_impl(hardware="B200", point="898896cd")
@oracle_impl(hardware="B200", point="59475c6e")
@oracle_impl(hardware="B200", point="1098a0cd")
@oracle_impl(hardware="B200", point="6844a365")
@oracle_impl(hardware="B200", point="d94b56b7")
@oracle_impl(hardware="B200", point="e8722254")
@oracle_impl(hardware="B200", point="d7001ca8")
@oracle_impl(hardware="B200", point="1adafdac")
@oracle_impl(hardware="B200", point="f054831c")
@oracle_impl(hardware="B200", point="a9bad761")
@oracle_impl(hardware="B200", point="d4748368")
@oracle_impl(hardware="B200", point="320fe2b8")
@oracle_impl(hardware="B200", point="f8f31da1")
@oracle_impl(hardware="B200", point="12c4ea37")
def oracle_forward(inputs):
    weight, gain, _view_shape, out_shape_param = inputs
    channels = int(weight.shape[0])
    inner_size = int(weight.numel() // channels)
    out_shape = _as_shape(out_shape_param)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=weight.device,
        dtype=torch.bfloat16,
    )
    inner_pad = _next_pow2(inner_size)

    weight_2d = weight.view(channels, inner_size)
    out_2d = out.view(channels, inner_size)
    gain_1d = gain.view(channels)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, 1, 1),
        _channel_norm_scale_bf16_kernel,
        (weight_2d, gain_1d, out_2d, inner_size, inner_pad),
    )
    return out
