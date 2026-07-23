"""cuTile port of sum_281f473c7326: NFNet-style GELU-derivative + channel sum.

Reference (channels-last input [N, C, H, W]):
  scaled = lhs * 1.7015043497085571 (bf16 chain)
  cdf = (erf(rhs * sqrt(1/2)) + 1) * 0.5
  pdf = rhs * exp(-0.5 * rhs^2) * (1/sqrt(2*pi))
  out_bf16 = bf16(scaled_f32 * (cdf + pdf))
  sum_channels = sum over (n, h, w) of out_bf16

cuTile doesn't have `erf` so we precompute `cdf + pdf` in torch, then run cuTile
kernels for the multiplication and channel reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


GAMMA = 1.7015043497085571


@ct.kernel
def _materialize_partial_kernel(
    scaled_ptr,       # bf16 [M, C]  (channels-last flattened)
    factor_ptr,       # f32 [M, C]  = cdf + pdf
    out_ptr,          # bf16 [M, C]
    partial_ptr,      # f32 [num_row_tiles, C]
    M: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    scaled = ct.load(
        scaled_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    factor = ct.load(
        factor_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scaled_f = ct.astype(scaled, ct.float32)
    value = scaled_f * factor
    value_bf16 = ct.astype(value, ct.bfloat16)
    ct.store(out_ptr, index=(row_tile, col_tile), tile=value_bf16)

    row_idx = ct.arange(BLOCK_M, dtype=ct.int32) + row_tile * BLOCK_M
    active = ct.reshape(row_idx < M, (BLOCK_M, 1))
    contrib = ct.where(active, ct.astype(value_bf16, ct.float32), 0.0)
    partial = ct.sum(contrib, axis=0)
    ct.store(partial_ptr, index=(row_tile, col_tile), tile=ct.reshape(partial, (1, BLOCK_C)))


@ct.kernel
def _finalize_channel_sum_kernel(
    partial_ptr,       # f32 [num_row_tiles, C]
    out_ptr,           # f32 [C]
    NUM_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    ROW_TILE_BLOCK: ct.Constant[int],
):
    col_tile = ct.bid(0)
    tile = ct.load(
        partial_ptr, index=(0, col_tile), shape=(ROW_TILE_BLOCK, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_row = ct.arange(ROW_TILE_BLOCK, dtype=ct.int32)
    row_mask = ct.reshape(tile_row < NUM_TILES, (ROW_TILE_BLOCK, 1))
    valid = ct.where(row_mask, tile, 0.0)
    total = ct.sum(valid, axis=0)
    total_bf16 = ct.astype(total, ct.bfloat16)
    total_f32 = ct.astype(total_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=total_f32)


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


def _dispatch(inputs):
    lhs, rhs = inputs
    device = lhs.device
    n = int(lhs.shape[0])
    c = int(lhs.shape[1])
    h = int(lhs.shape[2])
    w = int(lhs.shape[3])

    # Channels-last layout means storage order is (n, h, w, c).
    # We work with the transposed logical view [M=N*H*W, C] contiguous.
    lhs_mc = lhs.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)
    rhs_mc = rhs.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)

    # Precompute cdf + pdf as f32 [M, C]. erf/exp aren't in cuTile, and this
    # keeps rounding identical to eager for those transcendentals.
    rhs_f = rhs_mc.to(torch.float32)
    cdf = (torch.erf(rhs_f * 0.7071067811865476) + 1.0) * 0.5
    pdf = rhs_f * torch.exp(rhs_f * rhs_f * -0.5) * 0.3989422804014327
    factor = cdf + pdf  # f32 [M, C]

    # scaled = bf16(lhs * 1.7015...) — one bf16 rounding boundary in eager.
    scaled = (lhs.to(torch.float32) * GAMMA).to(torch.bfloat16)
    scaled_mc = scaled.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)

    M = n * h * w
    BLOCK_M = 128
    BLOCK_C = _next_p2(c)  # pad channel dim if not power-of-2
    num_row_tiles = (M + BLOCK_M - 1) // BLOCK_M

    # Round M up to BLOCK_M so the (M, C) tile can be exactly tiled. Pad the
    # inputs with zeros where needed.
    padded_M = num_row_tiles * BLOCK_M
    if padded_M != M:
        pad_rows = padded_M - M
        scaled_pad = torch.cat([
            scaled_mc,
            torch.zeros(pad_rows, c, device=device, dtype=torch.bfloat16),
        ], dim=0)
        factor_pad = torch.cat([
            factor,
            torch.zeros(pad_rows, c, device=device, dtype=torch.float32),
        ], dim=0)
    else:
        scaled_pad = scaled_mc
        factor_pad = factor

    # Pad channel dim too if C isn't power-of-2.
    if BLOCK_C != c:
        pad_cols = BLOCK_C - c
        scaled_pad = torch.cat([
            scaled_pad,
            torch.zeros(scaled_pad.shape[0], pad_cols, device=device, dtype=torch.bfloat16),
        ], dim=1)
        factor_pad = torch.cat([
            factor_pad,
            torch.zeros(factor_pad.shape[0], pad_cols, device=device, dtype=torch.float32),
        ], dim=1)

    out_pad = torch.empty((padded_M, BLOCK_C), device=device, dtype=torch.bfloat16)
    partial = torch.empty(
        (num_row_tiles, BLOCK_C), device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_tiles, 1, 1), _materialize_partial_kernel,
        (scaled_pad, factor_pad, out_pad, partial, M, c, BLOCK_M, BLOCK_C),
    )

    sum_out_pad = torch.empty((BLOCK_C,), device=device, dtype=torch.float32)
    ROW_TILE_BLOCK = _next_p2(num_row_tiles)
    ct.launch(
        stream, (1, 1, 1), _finalize_channel_sum_kernel,
        (partial, sum_out_pad, num_row_tiles, BLOCK_C, ROW_TILE_BLOCK),
    )

    # Rebuild channels-last output tensor from padded out_pad.
    out_channels_last = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, w * c, c),
        device=device,
        dtype=torch.bfloat16,
    )
    out_valid = out_pad.narrow(0, 0, M).narrow(1, 0, c).view(n, h, w, c)
    out_channels_last.copy_(out_valid.permute(0, 3, 1, 2))

    sum_out = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    sum_out.copy_(sum_out_pad.narrow(0, 0, c))

    return out_channels_last, sum_out


@oracle_impl(hardware="B200", point="75a244e4")
@oracle_impl(hardware="B200", point="9fd4ebb8")
@oracle_impl(hardware="B200", point="8bfea79e")
@oracle_impl(hardware="B200", point="d0c6a4b4")
@oracle_impl(hardware="B200", point="1b161b98")
@oracle_impl(hardware="B200", point="bd252f89")
def oracle_forward(inputs, **_kwargs):
    return _dispatch(inputs)
