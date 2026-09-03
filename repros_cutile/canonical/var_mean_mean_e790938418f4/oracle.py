"""cuTile port of var_mean_mean_e790938418f4: EfficientNet BN-training + SiLU + spatial mean.

Four-kernel design mirroring the Triton reference (partial+finalize+silu+mean):
  (1) `_partial_stats_kernel` — per-channel per-chunk sum / sum-of-squares.
  (2) `_finalize_stats_kernel` — reduces partials to mean/invstd; updates running stats.
  (3) `_silu_kernel` — BN affine + bf16 round + SiLU (elementwise).
  (4) `_spatial_mean_kernel` — mean over HW of the SiLU output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
EPS = 1.0e-3
VAR_CORRECTION = 1.0001594642002871


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_stats_kernel(
    x_ptr,             # bf16 [N*C, HW_TILES_PER_ROW, HW_TILE]
    partial_sum_ptr,   # f32 [N*C, chunks_per_row]
    partial_sum2_ptr,  # f32 [N*C, chunks_per_row]
    HW: ct.Constant[int],
    HW_TILE: ct.Constant[int],
    CHUNKS_PER_ROW: ct.Constant[int],
    N_C: ct.Constant[int],
):
    """One program per (row=n*C+c, hw-chunk).

    x is (N*C, HW). Each program reduces HW_TILE elements at (row, chunk*HW_TILE:).
    """
    row = ct.bid(0)
    chunk = ct.bid(1)

    tile = ct.load(x_ptr, index=(row, chunk), shape=(1, HW_TILE),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_1d = ct.reshape(tile, (HW_TILE,))
    tile_f = ct.astype(tile_1d, ct.float32)

    # Mask hw positions >= HW.
    cols = ct.arange(HW_TILE, dtype=ct.int32) + chunk * HW_TILE
    valid = cols < HW
    zero_hw = ct.zeros((HW_TILE,), dtype=ct.float32)
    tile_v = ct.where(valid, tile_f, zero_hw)
    s = ct.sum(tile_v)
    s2 = ct.sum(tile_v * tile_v)
    ct.store(partial_sum_ptr, index=(row, chunk),
             tile=ct.reshape(ct.full((1,), s, dtype=ct.float32), (1, 1)))
    ct.store(partial_sum2_ptr, index=(row, chunk),
             tile=ct.reshape(ct.full((1,), s2, dtype=ct.float32), (1, 1)))


@ct.kernel
def _finalize_stats_kernel(
    partial_sum_ptr,     # f32 [N*C, chunks_per_row]  (viewed also as [N, C, chunks])
    partial_sum2_ptr,    # f32 [N*C, chunks_per_row]
    running_mean_ptr,    # f32 [C]
    running_var_ptr,     # f32 [C]
    mean_out_ptr,        # f32 [C]
    invstd_out_ptr,      # f32 [C]
    C: ct.Constant[int],
    N: ct.Constant[int],
    N_PAD: ct.Constant[int],
    CHUNKS_PER_ROW: ct.Constant[int],
    CHUNKS_PAD: ct.Constant[int],
    E: ct.Constant[int],
    VAR_CORRECTION_: ct.Constant[float],
):
    """One program per channel c. Reduces all rows-with-this-channel and all
    hw chunks.

    partial_sum shape is (N*C, chunks_per_row). For channel c, the row indices
    are c, c+C, c+2C, ..., c+(N-1)*C. We load a (N_PAD, CHUNKS_PAD) tile with a
    stride-based indexing.
    """
    c = ct.bid(0)

    # Iterate over n (using static_iter is not scalable). Instead, treat
    # partial_sum as [N, C, CHUNKS_PER_ROW] and load one column (fixed c).
    # But partial_sum is 2D. So load column c stride-wise via loop over n.
    total = 0.0
    total2 = 0.0
    for n in ct.static_iter(range(N)):
        row = n * C + c
        p_sum = ct.load(partial_sum_ptr, index=(row, 0),
                        shape=(1, CHUNKS_PAD),
                        padding_mode=ct.PaddingMode.ZERO)
        p_sum2 = ct.load(partial_sum2_ptr, index=(row, 0),
                         shape=(1, CHUNKS_PAD),
                         padding_mode=ct.PaddingMode.ZERO)
        p_sum_1d = ct.reshape(p_sum, (CHUNKS_PAD,))
        p_sum2_1d = ct.reshape(p_sum2, (CHUNKS_PAD,))
        idx = ct.arange(CHUNKS_PAD, dtype=ct.int32)
        valid = idx < CHUNKS_PER_ROW
        zero = ct.zeros((CHUNKS_PAD,), dtype=ct.float32)
        p_sum_v = ct.where(valid, p_sum_1d, zero)
        p_sum2_v = ct.where(valid, p_sum2_1d, zero)
        total = total + ct.sum(p_sum_v)
        total2 = total2 + ct.sum(p_sum2_v)

    inv_e = 1.0 / E
    mean = total * inv_e
    ex2 = total2 * inv_e
    var_raw = ex2 - mean * mean
    var = ct.where(var_raw < 0.0, 0.0, var_raw)
    invstd = ct.rsqrt(var + EPS)

    ct.store(mean_out_ptr, index=(c,),
             tile=ct.reshape(ct.full((1,), mean, dtype=ct.float32), (1,)))
    ct.store(invstd_out_ptr, index=(c,),
             tile=ct.reshape(ct.full((1,), invstd, dtype=ct.float32), (1,)))

    old_mean = ct.load(running_mean_ptr, index=(c,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(c,), shape=(1,))
    mean_t = ct.reshape(ct.full((1,), mean, dtype=ct.float32), (1,))
    var_t = ct.reshape(ct.full((1,), var, dtype=ct.float32), (1,))
    new_mean = mean_t * 0.1 + old_mean * 0.9
    corrected = var_t * VAR_CORRECTION_
    new_var = corrected * 0.1 + old_var * 0.9
    ct.store(running_mean_ptr, index=(c,), tile=new_mean)
    ct.store(running_var_ptr, index=(c,), tile=new_var)


@ct.kernel
def _silu_kernel(
    x_ptr,          # bf16 [N*C, HW]
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    out_ptr,        # bf16 [N*C, HW]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_TILE: ct.Constant[int],
):
    """One program per (row=n*C+c, hw-chunk)."""
    row = ct.bid(0)
    chunk = ct.bid(1)
    c = row - (row // C) * C

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))

    tile = ct.load(x_ptr, index=(row, chunk), shape=(1, HW_TILE),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_1d = ct.astype(ct.reshape(tile, (HW_TILE,)), ct.float32)
    centered = tile_1d - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf, ct.float32)
    neg = affine_f * (-1.0)
    denom = ct.exp(neg) + 1.0
    silu = affine_f / denom
    silu_bf = ct.astype(silu, ct.bfloat16)
    ct.store(out_ptr, index=(row, chunk),
             tile=ct.reshape(silu_bf, (1, HW_TILE)))


@ct.kernel
def _spatial_mean_kernel(
    silu_ptr,        # bf16 [N*C, HW]
    out_ptr,         # bf16 [N*C]
    HW: ct.Constant[int],
    HW_TILE: ct.Constant[int],
    CHUNKS_PER_ROW: ct.Constant[int],
):
    """One program per (n, c) row. Reduces across HW using accumulator over chunks."""
    row = ct.bid(0)
    acc = 0.0
    for chunk in ct.static_iter(range(CHUNKS_PER_ROW)):
        tile = ct.load(silu_ptr, index=(row, chunk), shape=(1, HW_TILE),
                       padding_mode=ct.PaddingMode.ZERO)
        tile_1d = ct.astype(ct.reshape(tile, (HW_TILE,)), ct.float32)
        cols = ct.arange(HW_TILE, dtype=ct.int32) + chunk * HW_TILE
        valid = cols < HW
        zero_hw = ct.zeros((HW_TILE,), dtype=ct.float32)
        tile_v = ct.where(valid, tile_1d, zero_hw)
        acc = acc + ct.sum(tile_v)
    mean = acc * (1.0 / HW)
    mean_bf = ct.astype(mean, ct.bfloat16)
    ct.store(out_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), mean_bf, dtype=ct.bfloat16), (1,)))


@oracle_impl(hardware="B200", point="c2791544", CHANNELS=1152, HEIGHT=7, WIDTH=7)
@oracle_impl(hardware="B200", point="dc4c8729", CHANNELS=672, HEIGHT=7, WIDTH=7)
@oracle_impl(hardware="B200", point="e41460a6", CHANNELS=672, HEIGHT=14, WIDTH=14)
@oracle_impl(hardware="B200", point="886f27c9", CHANNELS=480, HEIGHT=14, WIDTH=14)
@oracle_impl(hardware="B200", point="8ffa8a4f", CHANNELS=240, HEIGHT=14, WIDTH=14)
@oracle_impl(hardware="B200", point="cdca2f80", CHANNELS=240, HEIGHT=28, WIDTH=28)
@oracle_impl(hardware="B200", point="9d6ef1f8", CHANNELS=144, HEIGHT=28, WIDTH=28)
@oracle_impl(hardware="B200", point="46238279", CHANNELS=144, HEIGHT=56, WIDTH=56)
@oracle_impl(hardware="B200", point="21476974", CHANNELS=96, HEIGHT=56, WIDTH=56)
@oracle_impl(hardware="B200", point="bf1cc8fb", CHANNELS=32, HEIGHT=112, WIDTH=112)
def oracle_forward(inputs, *, CHANNELS: int, HEIGHT: int, WIDTH: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    device = arg0_1.device
    n = BATCH
    c = CHANNELS
    hw = HEIGHT * WIDTH
    e = n * hw

    # Pick HW_TILE such that per-tile shape is manageable.
    # For hw <= 1024, use HW_TILE = HW_PAD.
    # For larger hw, use HW_TILE=1024 and iterate chunks in kernel loops.
    if hw <= 1024:
        HW_TILE = _next_pow2(hw)
    elif hw <= 4096:
        HW_TILE = 1024
    else:
        HW_TILE = 1024
    chunks_per_row = (hw + HW_TILE - 1) // HW_TILE
    HW_PADDED = chunks_per_row * HW_TILE

    # x_flat (N*C, HW) contiguous.
    x_flat = arg0_1.permute(0, 1, 2, 3).contiguous().view(n * c, hw)

    # Allocations
    n_rows = n * c
    partial_sum = torch.zeros((n_rows, chunks_per_row), device=device, dtype=torch.float32)
    partial_sum2 = torch.zeros((n_rows, chunks_per_row), device=device, dtype=torch.float32)
    mean_out = torch.empty((c,), device=device, dtype=torch.float32)
    invstd_out = torch.empty((c,), device=device, dtype=torch.float32)
    silu_flat = torch.empty((n_rows, hw), device=device, dtype=torch.bfloat16)
    spatial_flat = torch.empty((n_rows,), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()

    # (1) partial stats
    ct.launch(
        stream, (n_rows, chunks_per_row, 1), _partial_stats_kernel,
        (x_flat, partial_sum, partial_sum2, hw, HW_TILE, chunks_per_row, n_rows),
    )
    # (2) finalize stats
    CHUNKS_PAD = _next_pow2(chunks_per_row)
    N_PAD = _next_pow2(n)
    ct.launch(
        stream, (c, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sum2, arg1_1, arg2_1, mean_out, invstd_out,
         c, n, N_PAD, chunks_per_row, CHUNKS_PAD, e, VAR_CORRECTION),
    )
    # (3) silu
    ct.launch(
        stream, (n_rows, chunks_per_row, 1), _silu_kernel,
        (x_flat, mean_out, invstd_out, arg3_1, arg4_1, silu_flat,
         c, hw, HW_TILE),
    )
    # (4) spatial mean
    ct.launch(
        stream, (n_rows, 1, 1), _spatial_mean_kernel,
        (silu_flat, spatial_flat, hw, HW_TILE, chunks_per_row),
    )

    # Assemble outputs.
    mean_val = mean_out.view(1, c, 1, 1)
    rsqrt_val = invstd_out.view(1, c, 1, 1)

    # silu output was strided channels-last; return in (n, c, h, w) matching x stride.
    silu_out = torch.empty_strided(tuple(arg0_1.shape), tuple(arg0_1.stride()),
                                     device=device, dtype=torch.bfloat16)
    silu_out.copy_(silu_flat.view(n, c, HEIGHT, WIDTH))

    mean_spatial_out = spatial_flat.view(n, c, 1, 1)

    return mean_val, rsqrt_val, silu_out, mean_spatial_out, arg1_1, arg2_1
