"""cuTile port of mean_3840584eef9a: GhostNet BN-inference affine + spatial mean.

Input x is bf16 channels-last [B, C, H, W]. Per (b, c) row, apply BN affine
(rounded to bf16), then compute the bf16 mean of the H*W spatial values.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_affine_spatial_mean_kernel(
    running_mean_ptr,   # bf16 [C]
    x_ptr,              # bf16 [rows=B*C, HW] flat contiguous
    running_var_ptr,    # bf16 [C]
    weight_ptr,         # bf16 [C]
    bias_ptr,           # bf16 [C]
    out_ptr,            # bf16 [rows=B*C, HW] flat contiguous
    mean_out_ptr,       # bf16 [rows=B*C] flat
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

    mean = ct.astype(ct.load(running_mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(running_var_ptr, index=(channel,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)

    invstd = ct.rsqrt(var + EPS)
    normalized = (x_f - mean) * invstd
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    # Mask out OOB elements for both store and mean.
    cols = ct.arange(HW_PAD, dtype=ct.int32)
    valid = cols < HW
    valid_2d = ct.reshape(valid, (1, HW_PAD))
    zero_bf16 = ct.full(shape=(1, HW_PAD), fill_value=0.0, dtype=ct.bfloat16)
    out_masked = ct.where(valid_2d, affine_bf16, zero_bf16)

    ct.store(out_ptr, index=(row, 0), tile=out_masked)

    mean_val = ct.sum(ct.astype(out_masked, ct.float32)) / HW_F
    mean_bf = ct.astype(mean_val, ct.bfloat16)
    ct.store(mean_out_ptr, index=(row,), tile=ct.reshape(mean_bf, (1,)))


def _launch(inputs, *, C: int, HW: int):
    running_mean, x, running_var, weight, bias = inputs
    batch = int(x.shape[0])
    height = int(x.shape[2])
    width = int(x.shape[3])

    # Output preserves x's shape and stride (channels-last).
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (batch, C, 1, 1),
        (C, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    rows = batch * C
    # x is bf16 channels-last: stride = (C*H*W, 1, W*C, C). To view as [rows=B*C, HW]
    # we'd have to permute. Instead pass x as-is with a permuted view for a
    # contiguous logical [B, C, H, W] shape.
    x_nchw = x.permute(0, 1, 2, 3).contiguous().view(rows, HW)
    out_nchw_buf = torch.empty_strided(
        (rows, HW),
        (HW, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_1d = mean.view(rows)
    HW_PAD = _next_pow2(HW)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bn_affine_spatial_mean_kernel,
        (running_mean, x_nchw, running_var, weight, bias, out_nchw_buf, mean_1d, HW, HW_PAD, float(HW), C),
    )
    # out_nchw_buf is [rows, HW] contiguous; reshape to (B, C, H, W) then permute to
    # channels-last layout matching x's stride.
    out_nchw = out_nchw_buf.view(batch, C, height, width)
    out.copy_(out_nchw)
    return out, mean


@oracle_impl(hardware="B200", point="57e42e70", C=672, HW=49)
@oracle_impl(hardware="B200", point="bddd3dfb", C=72, HW=784)
def oracle_forward(inputs, *, C: int, HW: int):
    return _launch(inputs, C=C, HW=HW)
