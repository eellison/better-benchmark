"""cuTile port of var_mean_mean_de769927fa58: MobileViT/EfficientNet training-BN + SiLU + spatial mean.

Two-kernel design mirroring the Triton reference:
  (1) `_partial_stats_kernel` — split-K sums / sums-of-squares per channel block.
  (2) `_finalize_silu_mean_kernel` — reduces partials to per-channel mean/invstd,
      updates running stats, then computes affine + bf16 round + SiLU + spatial mean
      per (n, c) row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
BLOCK_E = 4096  # matches Triton BLOCK_E
C_BLOCK = 16    # channel tile
ROW_BLOCK = 1   # one (n,c) row per program (simple mapping)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_stats_kernel(
    x_ptr,              # bf16 [rows_flat] laid out as [N*C, HW]
    partial_sum_ptr,    # f32 [num_chunks, C]
    partial_sum2_ptr,   # f32 [num_chunks, C]
    E: ct.Constant[int],
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    BLOCK_E_: ct.Constant[int],
    C: ct.Constant[int],
    C_BLOCK_: ct.Constant[int],
    N: ct.Constant[int],
):
    """One program handles a (chunk, c_block) tile of the per-channel sum.

    Layout trick: x_ptr is [N*C, HW]. For each channel c and chunk r_block,
    we want to sum x[n*C+c, hw] over n in [n_lo, n_hi) and hw in [0, HW).
    We enumerate elements as e = n * HW + hw, and load x[n, c, hw] via
    two-tile row indexing.
    """
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    c_start = c_block * C_BLOCK_
    e_start = r_block * BLOCK_E_

    # Build 2D tile [BLOCK_E_, C_BLOCK_] of x values.
    # e index -> (n, hw) via n = e // HW, hw = e % HW.
    # For each n, load a (1, C, HW_PAD) tile? Too big. Instead, load per-row:
    # rows in x_flat are indexed by n*C + c.
    # For each (channel_j, e_i), we want x[n_i, c_j, hw_i]. We use a two-loop
    # via loading BLOCK_E_ pairs of (n, hw), then C_BLOCK_ channels.
    #
    # Simpler: kernel handles ONE (c_block, r_block) pair. r_block spans
    # BLOCK_E_ consecutive e values -> assumes BLOCK_E_ divides HW * n_stride
    # or we can pad. Let's use a load that walks (n_within_chunk, hw) for
    # channels in [c_start, c_start+C_BLOCK_).

    # x_ptr has shape (N*C, HW) — 2D. Each row is one (n, c) pair, columns are hw.
    # For a chunk of BLOCK_E_ e's, that's ceil(BLOCK_E_/HW) n's per chunk.
    # For simplicity, assume BLOCK_E_ % HW == 0 -> N_PER_CHUNK = BLOCK_E_ // HW.
    # Then load a (N_PER_CHUNK, C_BLOCK_, HW_PAD) tile? Too big in general.
    # Simplest: iterate n_per_chunk inside kernel.

    # Precompute n_per_chunk and n_start_in_chunk.
    n_per_chunk = BLOCK_E_ // HW
    n_start = r_block * n_per_chunk

    zero = ct.zeros((C_BLOCK_,), dtype=ct.float32)
    s_sum = zero
    s_sum2 = zero
    hw_arange = ct.arange(HW_PAD, dtype=ct.int32)
    hw_mask_1d = hw_arange < HW  # (HW_PAD,)
    for c_local in ct.static_iter(range(C_BLOCK_)):
        c = c_start + c_local
        c_valid = c < C
        acc_sum = 0.0
        acc_sum2 = 0.0
        for n_local in ct.static_iter(range(n_per_chunk)):
            n = n_start + n_local
            row_idx = n * C + c
            # x is (N*C, HW). Load one row of HW_PAD (padded) elements.
            tile = ct.load(
                x_ptr, index=(row_idx, 0), shape=(1, HW_PAD),
                padding_mode=ct.PaddingMode.ZERO,
            )
            tile_1d = ct.reshape(tile, (HW_PAD,))
            tile_f = ct.astype(tile_1d, ct.float32)
            zero_hw = ct.zeros((HW_PAD,), dtype=ct.float32)
            tile_valid = ct.where(hw_mask_1d, tile_f, zero_hw)
            acc_sum = acc_sum + ct.sum(tile_valid)
            acc_sum2 = acc_sum2 + ct.sum(tile_valid * tile_valid)
        # Store the per-channel partial as one scalar.
        val = ct.where(c_valid,
                       ct.full((1,), acc_sum, dtype=ct.float32),
                       ct.zeros((1,), dtype=ct.float32))
        val2 = ct.where(c_valid,
                        ct.full((1,), acc_sum2, dtype=ct.float32),
                        ct.zeros((1,), dtype=ct.float32))
        # store scalar at (r_block, c)
        ct.store(partial_sum_ptr, index=(r_block, c),
                 tile=ct.reshape(val, (1, 1)))
        ct.store(partial_sum2_ptr, index=(r_block, c),
                 tile=ct.reshape(val2, (1, 1)))


@ct.kernel
def _finalize_and_silu_mean_kernel(
    x_ptr,               # bf16 [N*C, HW]
    running_mean_ptr,    # f32 [C]
    running_var_ptr,     # f32 [C]
    weight_ptr,          # f32 [C]
    bias_ptr,            # f32 [C]
    partial_sum_ptr,     # f32 [num_chunks, C]
    partial_sum2_ptr,    # f32 [num_chunks, C]
    saved_mean_ptr,      # f32 [C]
    invstd_ptr,          # f32 [C]
    spatial_mean_ptr,    # bf16 [N*C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    E: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    N: ct.Constant[int],
    VAR_CORRECTION: ct.Constant[float],
):
    """One program per (n, c) row. Also updates saved_mean/invstd/running once
    per channel (guarded on n==0)."""
    row = ct.bid(0)  # row = n * C + c
    c = row - (row // C) * C
    n = row // C

    # Load partials for this channel across all chunks and reduce.
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunks_mask = chunks < NUM_CHUNKS  # (BLOCK_CHUNKS,)
    # partial_sum shape (num_chunks, C); pull column c.
    p_sum = ct.load(partial_sum_ptr, index=(0, c), shape=(BLOCK_CHUNKS, 1),
                    padding_mode=ct.PaddingMode.ZERO)
    p_sum2 = ct.load(partial_sum2_ptr, index=(0, c), shape=(BLOCK_CHUNKS, 1),
                     padding_mode=ct.PaddingMode.ZERO)
    p_sum_1d = ct.reshape(p_sum, (BLOCK_CHUNKS,))
    p_sum2_1d = ct.reshape(p_sum2, (BLOCK_CHUNKS,))
    zero_bc = ct.zeros((BLOCK_CHUNKS,), dtype=ct.float32)
    p_sum_v = ct.where(chunks_mask, p_sum_1d, zero_bc)
    p_sum2_v = ct.where(chunks_mask, p_sum2_1d, zero_bc)
    total = ct.sum(p_sum_v)
    total2 = ct.sum(p_sum2_v)

    inv_e = 1.0 / E
    mean = total * inv_e
    ex2 = total2 * inv_e
    var_raw = ex2 - mean * mean
    var = ct.where(var_raw < 0.0, 0.0, var_raw)
    invstd = ct.rsqrt(var + EPS)

    # Only n==0 program updates the per-channel outputs.
    if n == 0:
        # Load and update running.
        old_mean = ct.load(running_mean_ptr, index=(c,), shape=(1,))
        old_var = ct.load(running_var_ptr, index=(c,), shape=(1,))
        new_mean_v = ct.reshape(
            ct.full((1,), mean * 0.1, dtype=ct.float32) + old_mean * 0.9,
            (1,))
        corrected_var = var * VAR_CORRECTION
        new_var_v = ct.reshape(
            ct.full((1,), corrected_var * 0.1, dtype=ct.float32) + old_var * 0.9,
            (1,))
        ct.store(running_mean_ptr, index=(c,), tile=new_mean_v)
        ct.store(running_var_ptr, index=(c,), tile=new_var_v)
        ct.store(saved_mean_ptr, index=(c,),
                 tile=ct.reshape(ct.full((1,), mean, dtype=ct.float32), (1,)))
        ct.store(invstd_ptr, index=(c,),
                 tile=ct.reshape(ct.full((1,), invstd, dtype=ct.float32), (1,)))

    # Load channel affine params.
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    w_scalar = weight * 1.0  # (1,)
    b_scalar = bias * 1.0

    # Load this row of x (HW_PAD-padded) and compute affine + bf16 round + SiLU.
    x = ct.load(x_ptr, index=(row, 0), shape=(1, HW_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    x_1d = ct.astype(ct.reshape(x, (HW_PAD,)), ct.float32)
    centered = x_1d - mean
    normalized = centered * invstd
    weight_v = ct.reshape(w_scalar, (1,))
    bias_v = ct.reshape(b_scalar, (1,))
    affine = normalized * weight_v + bias_v
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f2 = ct.astype(affine_bf, ct.float32)
    neg = affine_f2 * (-1.0)
    denom = ct.exp(neg) + 1.0
    silu_f = affine_f2 / denom
    silu_bf = ct.astype(silu_f, ct.bfloat16)
    silu_f2 = ct.astype(silu_bf, ct.float32)

    cols = ct.arange(HW_PAD, dtype=ct.int32)
    valid = cols < HW
    zero_hw = ct.zeros((HW_PAD,), dtype=ct.float32)
    silu_masked = ct.where(valid, silu_f2, zero_hw)
    pooled = ct.sum(silu_masked)
    mean_out = pooled * (1.0 / HW)
    mean_out_bf = ct.astype(mean_out, ct.bfloat16)
    ct.store(spatial_mean_ptr, index=(row,),
             tile=ct.reshape(mean_out_bf, (1,)))


@oracle_impl(hardware="B200", point="db4bc2f7")
@oracle_impl(hardware="B200", point="86b98b1a")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, _shape0, _shape1, _shape2 = inputs
    device = x.device
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    elements_per_channel = n * h * w
    hw = h * w
    var_correction = elements_per_channel / (elements_per_channel - 1.0)

    # Reshape x to (N*C, HW) contiguously by permute (channels-last -> NCHW).
    x_flat = x.permute(0, 1, 2, 3).contiguous().view(n * c, hw)
    HW_PAD = _next_pow2(hw)

    # Match Triton chunks: BLOCK_E over e = N * HW.
    # For simple layout, BLOCK_E must be a multiple of HW. Use n_per_chunk = ceil.
    # Simplest: 1 chunk covers full E if E <= BLOCK_E; else pick n_per_chunk.
    # For hw=64 (db4bc2f7) N=128 -> E=8192; BLOCK_E=4096 -> n_per_chunk=64, num_chunks=2.
    # For hw=49 (86b98b1a) N=128 -> E=6272; BLOCK_E=4096 -> hmm, 4096/49 not int.
    # Fall back: n_per_chunk = 1 => BLOCK_E = HW_PAD, num_chunks = N.
    # Simpler and robust: 1 chunk per n. Then BLOCK_E = HW_PAD*1, NUM_CHUNKS=N.
    n_per_chunk = 1
    BLOCK_E_ = HW_PAD * n_per_chunk
    num_chunks = n // n_per_chunk

    # Allocate partials
    partial_sum = torch.zeros((num_chunks, c), device=device, dtype=torch.float32)
    partial_sum2 = torch.zeros((num_chunks, c), device=device, dtype=torch.float32)
    saved_mean = torch.empty((c,), device=device, dtype=torch.float32)
    invstd = torch.empty((c,), device=device, dtype=torch.float32)
    spatial_mean_flat = torch.empty((n * c,), device=device, dtype=torch.bfloat16)

    # Round c up to pow2 for tile blocks.
    C_BLOCK_ = 16 if c % 16 == 0 else _next_pow2(min(c, 16))
    if c % C_BLOCK_ != 0:
        # fallback: pick a divisor
        C_BLOCK_ = 1
    grid_c = c // C_BLOCK_

    BLOCK_CHUNKS_ = _next_pow2(num_chunks)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (grid_c, num_chunks, 1),
        _partial_stats_kernel,
        (x_flat, partial_sum, partial_sum2,
         elements_per_channel, hw, HW_PAD, BLOCK_E_, c, C_BLOCK_, n),
    )
    ct.launch(
        stream,
        (n * c, 1, 1),
        _finalize_and_silu_mean_kernel,
        (x_flat, running_mean, running_var, weight, bias,
         partial_sum, partial_sum2, saved_mean, invstd, spatial_mean_flat,
         c, hw, HW_PAD, elements_per_channel, num_chunks, BLOCK_CHUNKS_, n,
         var_correction),
    )

    # Reshape outputs to eager shapes.
    saved_mean_out = saved_mean.view(1, c, 1, 1)
    invstd_out = invstd.view(1, c, 1, 1)
    view = spatial_mean_flat.view(n, c)
    return saved_mean_out, invstd_out, view, running_mean, running_var
