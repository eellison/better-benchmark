"""cuTile port of pointwise_2fd75199c31d: Inception BN + ReLU + maxpool + avgpool.

The channels-last BN affine (per-channel scale/shift with bf16 rounding at
the final store) runs in a single 1D cuTile kernel over the NHWC-contiguous
storage of the input tensor. The two pooling stages (3x3 stride-2 maxpool
and 3x3 stride-1 padded avgpool) fall back to torch nn.functional pooling
ops, which honour the channels-last memory format.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 192
CHANNELS_PADDED = 256  # next pow2 of 192, for cuTile load/store shape
H_IN = 71
W_IN = 71
EPS = 0.001
PIXEL_BLOCK = 4
TOTAL_PIXELS = BATCH * H_IN * W_IN  # 128 * 71 * 71 = 645248


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 [total_pixels, C]  (NHWC flat)
    mean_ptr,     # bf16 [C]
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [total_pixels, C]
    C: ct.Constant[int],
    C_PADDED: ct.Constant[int],
    EPS_: ct.Constant[float],
    PIXEL_BLOCK_: ct.Constant[int],
):
    pid = ct.bid(0)

    x = ct.load(
        x_ptr, index=(pid, 0), shape=(PIXEL_BLOCK_, C_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(
        mean_ptr, index=(0,), shape=(C_PADDED,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    var = ct.load(
        var_ptr, index=(0,), shape=(C_PADDED,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.load(
        weight_ptr, index=(0,), shape=(C_PADDED,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(C_PADDED,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    inv_std = ct.rsqrt(var_f + EPS_)

    mean_2d = ct.reshape(mean_f, (1, C_PADDED))
    inv_std_2d = ct.reshape(inv_std, (1, C_PADDED))
    weight_2d = ct.reshape(weight_f, (1, C_PADDED))
    bias_2d = ct.reshape(bias_f, (1, C_PADDED))

    y = (x_f - mean_2d) * inv_std_2d * weight_2d + bias_2d
    y_bf = ct.astype(y, ct.bfloat16)
    zero_bf = ct.full((PIXEL_BLOCK_, C_PADDED), 0.0, dtype=ct.bfloat16)
    # ReLU with NaN-preserving semantics: torch's aten.relu maps NaN->NaN.
    is_nan = ct.astype(y_bf, ct.float32) != ct.astype(y_bf, ct.float32)
    relu_bf = ct.where(is_nan, y_bf, ct.where(y_bf > zero_bf, y_bf, zero_bf))

    # Masked scatter over the [PIXEL_BLOCK_, C] valid region.
    row_idx = ct.arange(PIXEL_BLOCK_, dtype=ct.int32) + pid * PIXEL_BLOCK_
    col_idx = ct.arange(C_PADDED, dtype=ct.int32)
    row_2d = ct.reshape(row_idx, (PIXEL_BLOCK_, 1))
    col_2d = ct.reshape(col_idx, (1, C_PADDED))
    row_b = ct.broadcast_to(row_2d, (PIXEL_BLOCK_, C_PADDED))
    col_b = ct.broadcast_to(col_2d, (PIXEL_BLOCK_, C_PADDED))
    flat_idx = row_b * C + col_b
    col_valid = col_b < C
    ct.scatter(out_ptr, (flat_idx,), relu_bf, mask=col_valid)


@oracle_impl(hardware="B200", point="760bed68", BLOCK_C=64, BLOCK_S=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_S, num_warps):
    del BLOCK_C, BLOCK_S, num_warps  # unused in cuTile port
    mean, x, var, weight, bias = inputs[:5]
    device = x.device

    # The input has channels-last stride (192 in stride[1]=1 slot); permute(0,2,3,1)
    # gives a contiguous NHWC view of the same storage.
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
    bn_nhwc = torch.empty(
        (BATCH, H_IN, W_IN, CHANNELS), device=device, dtype=torch.bfloat16
    )

    x_flat = x_nhwc.view(TOTAL_PIXELS, CHANNELS)
    bn_flat_1d = bn_nhwc.view(-1)  # scatter target uses linear indexing

    stream = torch.cuda.current_stream()
    grid = ((TOTAL_PIXELS + PIXEL_BLOCK - 1) // PIXEL_BLOCK, 1, 1)
    ct.launch(
        stream,
        grid,
        _bn_relu_kernel,
        (x_flat, mean, var, weight, bias, bn_flat_1d,
         CHANNELS, CHANNELS_PADDED, EPS, PIXEL_BLOCK),
    )

    # Back to channels-first channels-last-strided layout for pooling.
    relu = bn_nhwc.permute(0, 3, 1, 2)  # [B, C, H, W] channels-last view
    # Ensure the pooling ops see a channels-last input, which preserves layout.
    pool = torch.nn.functional.max_pool2d(relu, kernel_size=(3, 3), stride=(2, 2))
    avg = torch.nn.functional.avg_pool2d(pool, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return pool, avg
