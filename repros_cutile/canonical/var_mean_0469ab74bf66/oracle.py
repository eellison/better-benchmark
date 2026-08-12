"""cuTile port of var_mean_0469ab74bf66: ConvNeXtV2 channel LayerNorm.

For each spatial row (batch, height, width), compute bf16 add(x0,x1) rounded
to bf16, then fp32 var/mean across channels with eps=1e-6 rsqrt, affine, and
cast back to bf16. C is not always a power of 2 (320, 160, 80), so we use
BLOCK_C = next_power_of_2(C) with padding=ZERO for load, and multiply the
per-channel centered values by a mask constant before the variance reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _channel_layernorm_add_bf16_kernel(
    x0_ptr, x1_ptr, weight_ptr, bias_ptr, out_ptr,
    C: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_tile = ct.bid(0)
    x0 = ct.astype(
        ct.load(x0_ptr, index=(row_tile, 0), shape=(BLOCK_ROWS, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    x1 = ct.astype(
        ct.load(x1_ptr, index=(row_tile, 0), shape=(BLOCK_ROWS, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    added_bf16 = ct.astype(x0 + x1, ct.bfloat16)
    values = ct.astype(added_bf16, ct.float32)

    # mask: 1.0 where channel < C, else 0.0
    ch_arange = ct.arange(BLOCK_C, dtype=ct.int32)
    ch_lim = ct.full(shape=(BLOCK_C,), fill_value=C, dtype=ct.int32)
    ch_valid = ch_arange < ch_lim
    ones_row = ct.full(shape=(BLOCK_C,), fill_value=1.0, dtype=ct.float32)
    zeros_row = ct.full(shape=(BLOCK_C,), fill_value=0.0, dtype=ct.float32)
    valid = ct.reshape(ct.where(ch_valid, ones_row, zeros_row),
                       (1, BLOCK_C))

    values_masked = values * valid

    inv_c = 1.0 / C
    sum_values = ct.sum(values_masked, axis=1, keepdims=True)
    mean = sum_values * ct.full(shape=(BLOCK_ROWS, 1),
                                 fill_value=inv_c, dtype=ct.float32)
    centered = (values - mean) * valid
    variance = (ct.sum(centered * centered, axis=1, keepdims=True)
                * ct.full(shape=(BLOCK_ROWS, 1),
                           fill_value=inv_c, dtype=ct.float32))
    invstd = ct.rsqrt(variance + ct.full(shape=(BLOCK_ROWS, 1),
                                          fill_value=1e-6,
                                          dtype=ct.float32))

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    output = centered * invstd * weight_2d + bias_2d
    ct.store(out_ptr, index=(row_tile, 0),
             tile=ct.astype(output, ct.bfloat16))


def _channels_last_stride(n, c, h, w):
    return (c * h * w, 1, c * w, c)


def _next_power_of_2(x):
    p = 1
    while p < x:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="274067f9", BLOCK_ROWS=16)
@oracle_impl(hardware="B200", point="f412d821", BLOCK_ROWS=16)
@oracle_impl(hardware="B200", point="29afe565", BLOCK_ROWS=32)
def oracle_forward(inputs, *, BLOCK_ROWS: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    total_rows = n * h * w
    BLOCK_C = _next_power_of_2(c)
    out = torch.empty_strided(
        (n, c, h, w), _channels_last_stride(n, c, h, w),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    # NHWC view (contiguous given channels-last input)
    x0_nhwc = arg0_1.permute(0, 2, 3, 1).view(total_rows, c)
    x1_nhwc = arg1_1.permute(0, 2, 3, 1).view(total_rows, c)
    out_nhwc = out.permute(0, 2, 3, 1).view(total_rows, c)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (total_rows // BLOCK_ROWS, 1, 1),
        _channel_layernorm_add_bf16_kernel,
        (x0_nhwc, x1_nhwc, arg2_1, arg3_1, out_nhwc,
         c, BLOCK_ROWS, BLOCK_C),
    )
    return out
