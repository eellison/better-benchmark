"""cuTile port of var_mean_c003cf6c87f4: ConvNeXt residual channel LayerNorm (C=80).

Row LayerNorm across 80 channels of a channels-last [N, C, H, W] input. We
view the input as NHWC (which matches its channels-last memory layout), so
the layernorm reduces over the innermost dim (C=80). C is not a power of 2:
we use BLOCK_C=128 with padding-zero load and column-masked reductions, and
we write to padded per-row outputs then slice back.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _channel_layernorm_kernel(
    x_ptr,          # bf16 [ROWS, C]  (NHWC-contiguous)
    residual_ptr,   # f32  [ROWS, C]
    weight_ptr,     # f32  [C]
    bias_ptr,       # f32  [C]
    norm_ptr,       # f32  [ROWS, BLOCK_C] (padded)
    out_ptr,        # bf16 [ROWS, BLOCK_C] (padded)
    div_ptr,        # f32  [ROWS]
    C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    values = ct.astype(x, ct.float32) + residual

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < C_C
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_C))
    zero = ct.full((1, BLOCK_C), 0.0, dtype=ct.float32)

    inv_c = 1.0 / C_C
    x_masked = ct.where(col_mask_2d, values, zero)
    mean = ct.sum(x_masked) * inv_c
    centered = values - mean
    centered_masked = ct.where(col_mask_2d, centered, zero)
    variance = ct.sum(centered_masked * centered_masked) * inv_c
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    affine = normalized * weight_2d + bias_2d
    out = ct.astype(affine, ct.bfloat16)
    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(out_ptr, index=(row, 0), tile=out)

    # scalar div store — length-1 tile.
    div_tile = ct.full((1,), 0.0, dtype=ct.float32) + invstd * inv_c
    ct.store(div_ptr, index=(row,), tile=div_tile)


@oracle_impl(hardware="B200", point="8b7d5a32", BLOCK_C=128, BLOCK_ROWS=32)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_ROWS: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n, c, h, w = (int(d) for d in arg0_1.shape)
    total_rows = n * h * w

    # Channels-last -> NHWC contiguous view (metadata-only for channels-last input).
    x_nhwc = arg0_1.permute(0, 2, 3, 1).reshape(total_rows, c)
    res_nhwc = arg1_1.permute(0, 2, 3, 1).reshape(total_rows, c)

    padded_norm = torch.empty(
        (total_rows, BLOCK_C),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    padded_out = torch.empty(
        (total_rows, BLOCK_C),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty(total_rows, device=arg0_1.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_rows, 1, 1),
        _channel_layernorm_kernel,
        (x_nhwc, res_nhwc, arg2_1, arg3_1, padded_norm, padded_out, div,
         c, BLOCK_C),
    )

    # normalized output is NHWC-contiguous — write into it via 1 strided copy.
    normalized = torch.empty_strided(
        (n, h, w, c),
        (h * w * c, w * c, c, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    normalized.view(total_rows, c).copy_(padded_norm[:, :c])

    # out is channels-last (n,c,h,w); its NHWC permute view is contiguous.
    out = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, c * w, c),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out.permute(0, 2, 3, 1).reshape(total_rows, c).copy_(padded_out[:, :c])

    div_out = torch.empty_strided(
        (n, h, w, 1),
        (h * w, w, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    div_out.view(total_rows).copy_(div)

    return normalized, out, div_out
