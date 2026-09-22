"""cuTile port of var_mean_e98d6d833b6e: ConvNeXt channel LayerNorm.

Input: bf16 [N, C, H, W] channels-last. For each (n, h, w) row of channels,
compute var+mean, rsqrt(var + 1e-6), affine (weight*normalized + bias),
final bf16 cast. Output mean [N, H, W, 1] f32, rsqrt [N, H, W, 1] f32,
output bf16 [N, C, H, W] channels-last.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _channel_layernorm_kernel(
    x_ptr,          # bf16 [ROWS, C] where ROWS = N*H*W
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    mean_ptr,       # f32 [ROWS]
    rsqrt_ptr,      # f32 [ROWS]
    out_ptr,        # bf16 [ROWS, C]
    C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    # Load one row, cast to f32
    x_bf16 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_C))
    x = ct.astype(x_bf16, ct.float32)
    sum_x = ct.sum(x)
    sum_x_sq = ct.sum(x * x)
    mean = sum_x * (1.0 / C)
    variance = sum_x_sq * (1.0 / C) - mean * mean
    variance = ct.maximum(variance, 0.0)
    centered = x - mean
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    affine = centered * invstd * weight_2d + bias_2d
    out = ct.astype(affine, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)

    # mean/rsqrt scalar per row - store as size-1 tile
    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))


@oracle_impl(hardware="B200", point="3162d0ee", BLOCK_C=1024)
@oracle_impl(hardware="B200", point="366664ec", BLOCK_C=512)
@oracle_impl(hardware="B200", point="bf60dc4e", BLOCK_C=256)
@oracle_impl(hardware="B200", point="32d9a8b7", BLOCK_C=128)
def oracle_forward(inputs, *, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    total_rows = n * h * w

    # arg0_1 is channels-last: stride = (C*H*W, 1, W*C, C). Permute back to
    # logical [N, H, W, C] — this is already contiguous, so reshape is
    # metadata-only (no copy).
    x_2d = arg0_1.permute(0, 2, 3, 1).reshape(total_rows, c)

    mean = torch.empty_strided(
        (n, h, w, 1),
        (h * w, w, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (n, h, w, 1),
        (h * w, w, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, c * w, c),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # Also view out as (rows, c). out has channels-last strides, so
    # permute(0,2,3,1) yields a contiguous [N,H,W,C] view we can reshape.
    out_2d_view = out.permute(0, 2, 3, 1).reshape(total_rows, c)

    mean_1d = mean.view(-1)
    rsqrt_1d = rsqrt.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_rows, 1, 1),
        _channel_layernorm_kernel,
        (x_2d, arg1_1, arg2_1, mean_1d, rsqrt_1d, out_2d_view, c, BLOCK_C),
    )
    return mean, rsqrt, out
