"""cuTile port of sum_sum_c17d298e608e: NFNet weight-standardization backward.

Per output-channel: compute weight-standardization normalized gradient. Handles
the channels-last-like input strides via torch's `.contiguous()` clone, then
runs the reduction+epilogue kernel with padded REDUCE_N.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


WEIGHT_SCALE = 0.19245008972987526


@ct.kernel
def _nfnet_bwd_kernel(
    grad_ptr,
    weight_ptr,
    mean_ptr,
    invstd_ptr,
    gain_ptr,
    out_gain_ptr,
    out_weight_ptr,
    REDUCE_N: ct.Constant[int],
    REDUCE_N_PAD: ct.Constant[int],
    INV_REDUCE_N: ct.Constant[float],
):
    oc = ct.bid(0)

    grad = ct.load(
        grad_ptr,
        index=(oc, 0),
        shape=(1, REDUCE_N_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.load(
        weight_ptr,
        index=(oc, 0),
        shape=(1, REDUCE_N_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    grad = ct.reshape(grad, (REDUCE_N_PAD,))
    weight = ct.reshape(weight, (REDUCE_N_PAD,))
    cols = ct.arange(REDUCE_N_PAD, dtype=ct.int32)
    col_mask = cols < REDUCE_N

    mean_tile = ct.load(mean_ptr, index=(oc,), shape=(1,))
    invstd_tile = ct.load(invstd_ptr, index=(oc,), shape=(1,))
    gain_tile = ct.load(gain_ptr, index=(oc,), shape=(1,))
    mean = ct.reshape(mean_tile, ())
    invstd = ct.reshape(invstd_tile, ())
    gain = ct.reshape(gain_tile, ())

    centered = weight - mean
    grad_masked = ct.where(col_mask, grad, 0.0)
    centered_masked = ct.where(col_mask, centered, 0.0)
    sum_grad = ct.sum(grad_masked)
    sum_grad_centered = ct.sum(grad_masked * centered_masked)

    mean_grad = sum_grad * INV_REDUCE_N
    invstd_sq = invstd * invstd
    variance_term = (sum_grad_centered * INV_REDUCE_N) * invstd_sq
    output_scale = invstd * (gain * WEIGHT_SCALE)

    tensor_out = (grad - centered * variance_term - mean_grad) * output_scale
    gain_out = (sum_grad_centered * invstd) * WEIGHT_SCALE

    ct.store(out_gain_ptr, index=(oc,), tile=ct.reshape(gain_out, (1,)))
    tensor_out_2d = ct.reshape(tensor_out, (1, REDUCE_N_PAD))
    ct.store(out_weight_ptr, index=(oc, 0), tile=tensor_out_2d)


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n & (n - 1) else n


def _launch(inputs):
    (
        grad,
        weight,
        mean,
        invstd,
        gain,
        _shape_param_0,
        _shape_param_1,
        out_gain_shape,
        out_weight_shape,
    ) = inputs

    out_channels = int(grad.shape[0])
    in_channels = int(grad.shape[1])
    reduce_n = in_channels * 9
    reduce_n_pad = _next_pow2(reduce_n)
    device = grad.device
    # NOTE: The Repro hardcodes INV_REDUCE_N = 1/27 = 0.037037037037037035
    # regardless of shape (captured trace from IC=3), so we mirror that.
    inv_reduce_n_hardcoded = 0.037037037037037035

    grad_flat = grad.to(torch.float32).contiguous().view(out_channels, reduce_n)
    weight_flat = weight.contiguous().view(out_channels, reduce_n)

    out_gain = torch.empty_strided(
        tuple(int(dim) for dim in out_gain_shape),
        (1, 1, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    out_weight_padded = torch.empty(
        (out_channels, reduce_n_pad),
        device=device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (out_channels, 1, 1),
        _nfnet_bwd_kernel,
        (
            grad_flat,
            weight_flat,
            mean.view(-1),
            invstd,
            gain.view(-1),
            out_gain.view(-1),
            out_weight_padded,
            reduce_n,
            reduce_n_pad,
            inv_reduce_n_hardcoded,
        ),
    )
    out_weight_flat = out_weight_padded[:, :reduce_n].contiguous()
    out_weight = out_weight_flat.view(
        tuple(int(dim) for dim in out_weight_shape)
    )
    return out_gain, out_weight


@oracle_impl(hardware="B200", point="dbe7e0fa")
@oracle_impl(hardware="B200", point="a7e28603")
@oracle_impl(hardware="B200", point="64510e66")
@oracle_impl(hardware="B200", point="3005b63c")
@oracle_impl(hardware="B200", point="442ee97d")
@oracle_impl(hardware="B200", point="8b771129")
@oracle_impl(hardware="B200", point="a1b0079b")
@oracle_impl(hardware="B200", point="77446b6f")
@oracle_impl(hardware="B200", point="f88f9f06")
def oracle_forward(inputs):
    return _launch(inputs)
