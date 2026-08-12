"""cuTile port of var_mean_05c4919007f9: ConvNeXtV2 channel-LayerNorm training.

One kernel processes BLOCK_ROWS rows x BLOCK_C channels per program,
mirroring Triton's ROW_BLOCK-tiled row kernel: NHWC residual add + rounded
bf16 store + channel var/mean + rsqrt side-outputs + affine bf16 out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _convnextv2_channel_layernorm_kernel(
    x0_ptr,          # bf16 [total_rows, C] (NHWC flat)
    x1_ptr,          # bf16 [total_rows, C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    add_nhwc_ptr,    # bf16 [total_rows, C]
    mean_ptr,        # f32 [total_rows]
    rsqrt_ptr,       # f32 [total_rows]
    out_bf16_ptr,    # bf16 [total_rows, C]
    C: ct.Constant[int],
    TOTAL_ROWS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
):
    row_block = ct.bid(0)

    x0_bf = ct.load(x0_ptr, index=(row_block, 0), shape=(BLOCK_ROWS, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    x1_bf = ct.load(x1_ptr, index=(row_block, 0), shape=(BLOCK_ROWS, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    values = ct.astype(x0_bf, ct.float32) + ct.astype(x1_bf, ct.float32)
    added = ct.astype(values, ct.bfloat16)

    row_ids = row_block * BLOCK_ROWS + ct.arange(BLOCK_ROWS, dtype=ct.int32)
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid_1d = row_ids < TOTAL_ROWS
    col_valid_1d = cols < C
    row_valid = ct.reshape(row_valid_1d, (BLOCK_ROWS, 1))
    col_valid = ct.reshape(col_valid_1d, (1, BLOCK_C))
    zero_b = ct.zeros((BLOCK_ROWS, BLOCK_C), dtype=ct.bool_)
    mask = (row_valid & col_valid) | zero_b

    ct.store(add_nhwc_ptr, index=(row_block, 0), tile=added)

    values_masked = ct.where(mask, values, 0.0)
    sum_values = ct.sum(values_masked, axis=1, keepdims=True)
    sum_squares = ct.sum(values_masked * values_masked, axis=1, keepdims=True)
    mean = sum_values / C
    variance = sum_squares / C - mean * mean
    variance = ct.where(variance > 0.0, variance,
                        ct.zeros((BLOCK_ROWS, 1), dtype=ct.float32))
    invstd = ct.rsqrt(variance + 1.0e-6)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    rounded_centered = ct.astype(added, ct.float32) - mean
    normalized = rounded_centered * invstd
    affine = normalized * weight_2d + bias_2d
    final = ct.astype(affine, ct.bfloat16)

    ct.store(mean_ptr, index=(row_block,),
             tile=ct.reshape(mean, (BLOCK_ROWS,)))
    ct.store(rsqrt_ptr, index=(row_block,),
             tile=ct.reshape(invstd, (BLOCK_ROWS,)))
    ct.store(out_bf16_ptr, index=(row_block, 0), tile=final)


def _channels_last_stride(batch, channels, height, width):
    return (channels * height * width, 1, channels * width, channels)


def _nhwc_stride(batch, height, width, channels):
    return (height * width * channels, width * channels, channels, 1)


@oracle_impl(hardware="B200", point="f9c4eb2d", BLOCK_C=512, BLOCK_ROWS=4)
@oracle_impl(hardware="B200", point="e5e4b0b5", BLOCK_C=256, BLOCK_ROWS=16)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_ROWS):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    total_rows = batch * height * width

    add_nhwc = torch.empty_strided(
        (batch, height, width, channels),
        _nhwc_stride(batch, height, width, channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (batch, height, width, 1),
        (height * width, width, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (batch, height, width, 1),
        (height * width, width, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (batch, channels, height, width),
        _channels_last_stride(batch, channels, height, width),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # channels-last strides -> physical layout is NHWC. Metadata-only 2D
    # views (no copies).
    x0_2d = torch.as_strided(arg0_1, (total_rows, channels), (channels, 1))
    x1_2d = torch.as_strided(arg1_1, (total_rows, channels), (channels, 1))
    add_nhwc_2d = add_nhwc.view(total_rows, channels)
    out_bf16_2d = torch.as_strided(out_bf16, (total_rows, channels),
                                   (channels, 1))
    mean_1d = mean.view(total_rows)
    rsqrt_1d = rsqrt.view(total_rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total_rows, BLOCK_ROWS), 1, 1),
        _convnextv2_channel_layernorm_kernel,
        (x0_2d, x1_2d, arg2_1, arg3_1, add_nhwc_2d, mean_1d, rsqrt_1d,
         out_bf16_2d, channels, total_rows, BLOCK_C, BLOCK_ROWS),
    )
    return add_nhwc, mean, rsqrt, out_bf16
