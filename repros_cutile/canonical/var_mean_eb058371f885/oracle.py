"""cuTile port of var_mean_eb058371f885: ConvNeXt channels-last LayerNorm over C.

Reduce along channels (non-power-of-2). Load with ZERO padding, mask-out OOB
elements before reduction, and use BLOCK_ROWS=1 so per-tile writes stay in-bounds.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _channel_ln_kernel(
    x_ptr,           # bf16 [TOTAL_ROWS, C]
    weight_ptr,      # bf16 [C]
    bias_ptr,        # bf16 [C]
    out_ptr,         # bf16 [TOTAL_ROWS, C]
    C: ct.Constant[int],
    C_PAD: ct.Constant[int],
    C_F: ct.Constant[float],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, C_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    cols = ct.arange(C_PAD, dtype=ct.int32)
    valid = cols < C
    valid_2d = ct.reshape(valid, (1, C_PAD))
    zero_f = ct.full(shape=(1, C_PAD), fill_value=0.0, dtype=ct.float32)

    x_masked = ct.where(valid_2d, x_f, zero_f)
    mean = ct.sum(x_masked) / C_F
    centered = x_f - mean
    centered_masked = ct.where(valid_2d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) / C_F
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(C_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(C_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, C_PAD))
    bias_2d = ct.reshape(bias_f, (1, C_PAD))

    out = centered * invstd * weight_2d + bias_2d
    out_bf16 = ct.astype(out, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out_bf16)


@oracle_impl(hardware="B200", point="e37cf831")
@oracle_impl(hardware="B200", point="285e3478")
@oracle_impl(hardware="B200", point="6b80bcdb")
@oracle_impl(hardware="B200", point="c5b90479")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    out = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, c * w, c),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    total_rows = n * h * w
    C_PAD = _next_pow2(c)
    # Input is channels-last (strides c*h*w, 1, c*w, c) — as [N, H, W, C]
    # linearly, it's [total_rows, c] contiguous.
    x_2d = arg0_1.permute(0, 2, 3, 1).reshape(total_rows, c)
    out_2d = out.permute(0, 2, 3, 1).view(total_rows, c)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_rows, 1, 1),
        _channel_ln_kernel,
        (x_2d, arg1_1, arg2_1, out_2d, c, C_PAD, float(c)),
    )
    return out
