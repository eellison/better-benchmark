"""cuTile port of var_mean_b48155c1deba: ConvNeXtV2 channel LayerNorm.

Normalize over channel dim (C=80), then affine, produce f32 mean/invstd,
f32 affine result, and bf16 result. Input is channels-last bf16.

Match Triton's BLOCK_R=8 to reduce launch overhead vs the naive
(N, H, W) grid.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _channel_ln_kernel(
    x_2d,         # (rows, C=80) bf16 - NHWC flattened
    weight,       # (C,) f32
    bias,         # (C,) f32
    mean_out_1d,  # (rows,) f32
    invstd_out_1d, # (rows,) f32
    out_f32_2d,   # (rows, C=80) f32
    out_bf16_2d,  # (rows, C=80) bf16
    ROWS: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    x = ct.load(x_2d, index=(row_block, 0), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    ch_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    ch_lt_c = ch_idx < ct.full(shape=(BLOCK_C,), fill_value=C, dtype=ct.int32)
    zero_2d = ct.zeros(shape=(BLOCK_R, BLOCK_C), dtype=ct.float32)
    ch_lt_c_2d = ct.reshape(ch_lt_c, (1, BLOCK_C))
    x_active = ct.where(ch_lt_c_2d, x_f, zero_2d)

    inv_c = 1.0 / C
    mean = ct.sum(x_active, axis=1) * inv_c  # (BLOCK_R,)
    mean_2d = ct.reshape(mean, (BLOCK_R, 1))
    centered = x_f - mean_2d
    centered_active = ct.where(ch_lt_c_2d, centered, zero_2d)
    variance = ct.sum(centered_active * centered_active, axis=1) * inv_c
    invstd = ct.rsqrt(variance + 1.0e-6)
    invstd_2d = ct.reshape(invstd, (BLOCK_R, 1))

    scale = ct.load(weight, index=(0,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    shift = ct.load(bias, index=(0,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    scale_2d = ct.reshape(scale, (1, BLOCK_C))
    shift_2d = ct.reshape(shift, (1, BLOCK_C))
    normalized = centered_active * invstd_2d
    affine = normalized * scale_2d + shift_2d

    # Store mean/invstd: rows dim divides ROWS since ROWS is a multiple of
    # BLOCK_R for our shape (128*56*56=401408, 401408 % 8 = 0).
    ct.store(mean_out_1d, index=(row_block,), tile=mean)
    ct.store(invstd_out_1d, index=(row_block,), tile=invstd)

    # Store affine: use scatter to avoid OOB channel writes since BLOCK_C=128
    # > C=80. The rows dim is in-bounds (multiple of BLOCK_R).
    r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + row_block * BLOCK_R
    r_idx_2d = ct.reshape(r_idx, (BLOCK_R, 1))
    ch_idx_2d = ct.reshape(ch_idx, (1, BLOCK_C))
    r_idx_bcast = r_idx_2d + ct.zeros(shape=(BLOCK_R, BLOCK_C), dtype=ct.int32)
    ch_idx_bcast = ch_idx_2d + ct.zeros(shape=(BLOCK_R, BLOCK_C), dtype=ct.int32)
    mask = ch_idx_bcast < ct.full(shape=(BLOCK_R, BLOCK_C), fill_value=C, dtype=ct.int32)
    ct.scatter(out_f32_2d, (r_idx_bcast, ch_idx_bcast), affine, mask=mask)
    ct.scatter(out_bf16_2d, (r_idx_bcast, ch_idx_bcast),
               ct.astype(affine, ct.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="32d9a8b7", BLOCK_R=8)
def oracle_forward(inputs, *, BLOCK_R: int):
    x, weight, bias = inputs
    N, C, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    rows = N * H * W
    # x's channels-last strides match NHWC contiguous strides: view as (rows, C)
    x_2d = torch.as_strided(x, (rows, C), (C, 1))

    out_strides = (C * H * W, 1, W * C, C)
    mean = torch.empty_strided(
        (N, H, W, 1), (H * W, W, 1, 1),
        device=x.device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (N, H, W, 1), (H * W, W, 1, 1),
        device=x.device, dtype=torch.float32,
    )
    out_f32 = torch.empty_strided(
        (N, C, H, W), out_strides,
        device=x.device, dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (N, C, H, W), out_strides,
        device=x.device, dtype=torch.bfloat16,
    )
    out_f32_2d = torch.as_strided(out_f32, (rows, C), (C, 1))
    out_bf16_2d = torch.as_strided(out_bf16, (rows, C), (C, 1))
    mean_1d = torch.as_strided(mean, (rows,), (1,))
    invstd_1d = torch.as_strided(invstd, (rows,), (1,))

    BLOCK_C = 128  # >= C=80, PoT
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(rows, BLOCK_R), 1, 1), _channel_ln_kernel,
              (x_2d, weight, bias, mean_1d, invstd_1d, out_f32_2d, out_bf16_2d,
               rows, C, BLOCK_R, BLOCK_C))
    return mean, invstd, out_f32, out_bf16
