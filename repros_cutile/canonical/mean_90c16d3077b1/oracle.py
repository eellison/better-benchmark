"""cuTile port of mean_90c16d3077b1: BN-inference + ReLU + spatial mean row kernel.

BN-inference-affine, explicit bf16 round, NaN-preserving ReLU, spatial mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    relu_out_ptr,
    mean_out_ptr,
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    HW_F: ct.Constant[float],
    C: ct.Constant[int],
):
    row = ct.bid(0)
    channel = row - (row // C) * C

    x = ct.load(
        x_ptr,
        index=(row, 0),
        shape=(1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    channel_mean = ct.astype(ct.load(mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    variance = ct.astype(ct.load(var_ptr, index=(channel,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)

    invstd = 1.0 / ct.sqrt(variance + 0.001)
    y = ((x_f - channel_mean) * invstd) * weight + bias
    y_bf16 = ct.astype(y, ct.bfloat16)
    zero_bf = ct.full(shape=(1, HW_PAD), fill_value=0.0, dtype=ct.bfloat16)
    relu = ct.where(y_bf16 < zero_bf, zero_bf, y_bf16)

    # Mask off padded positions before storing.
    cols = ct.arange(HW_PAD, dtype=ct.int32)
    valid_2d = ct.reshape(cols < HW, (1, HW_PAD))
    relu_masked = ct.where(valid_2d, relu, zero_bf)

    ct.store(relu_out_ptr, index=(row, 0), tile=relu_masked)

    mean_value = ct.sum(ct.astype(relu_masked, ct.float32)) / HW_F
    mean_bf = ct.astype(mean_value, ct.bfloat16)
    ct.store(mean_out_ptr, index=(row,), tile=ct.reshape(mean_bf, (1,)))


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _launch(inputs, *, BLOCK_HW: int):
    mean, x, variance, weight, bias = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    total_rows = batch * channels

    relu_out = torch.empty_strided(
        (batch, channels, height, width),
        (channels * hw, hw, width, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_out = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    x_2d = x.view(total_rows, hw)
    relu_2d = relu_out.view(total_rows, hw)
    mean_1d = mean_out.view(total_rows)
    hw_pad = _next_pow2(hw)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_rows, 1, 1),
        _bn_relu_mean_kernel,
        (mean, x_2d, variance, weight, bias, relu_2d, mean_1d, hw, hw_pad, float(hw), channels),
    )
    return (relu_out, mean_out)


@oracle_impl(hardware="B200", point="51b8f364", BLOCK_HW=1024)
@oracle_impl(hardware="B200", point="e78adaa5", BLOCK_HW=1024)
def oracle_forward(inputs, *, BLOCK_HW: int):
    return _launch(inputs, BLOCK_HW=BLOCK_HW)
