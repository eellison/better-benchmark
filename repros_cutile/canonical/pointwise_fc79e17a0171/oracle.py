"""cuTile port of pointwise_fc79e17a0171: DCGAN BN-inference affine + leaky ReLU.

Per-channel per-tile: normalize x with running mean/var, apply weight/bias, bf16 round,
then leaky_relu(x) = x if x > 0 else 0.2*x, and bf16 round back.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _bn_affine_leaky_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    # 2D contiguous view: (batch*channels, HW) tiled by (1, BLOCK_HW).
    row = ct.bid(0)  # 0..batch*channels
    channel = row - (row // CHANNELS) * CHANNELS

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    x_f = ct.astype(x, ct.float32)

    mean = ct.astype(ct.load(mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(channel,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)

    invstd = ct.rsqrt(var + EPS)
    centered = x_f - mean
    normalized = centered * invstd
    scaled = normalized * weight
    affine = scaled + bias
    rounded_bf16 = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf16, ct.float32)
    negative = rounded_f * 0.2
    zero = ct.full(shape=(1, BLOCK_HW), fill_value=0.0, dtype=ct.float32)
    out = ct.where(rounded_f > zero, rounded_f, negative)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="ade1975e", CHANNELS=512, HW=16)
@oracle_impl(hardware="B200", point="359745db", CHANNELS=256, HW=64)
@oracle_impl(hardware="B200", point="177337c2", CHANNELS=128, HW=256)
def oracle_forward(inputs, *, CHANNELS: int, HW: int):
    mean, x, var, weight, bias = inputs
    batch = int(x.shape[0])
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    rows = batch * CHANNELS
    # Reshape to (rows, HW) 2D contiguous views.
    x_2d = x.reshape(rows, HW)
    out_2d = out.reshape(rows, HW)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bn_affine_leaky_kernel,
        (mean, x_2d, var, weight, bias, out_2d, CHANNELS, HW, HW),
    )
    return out
