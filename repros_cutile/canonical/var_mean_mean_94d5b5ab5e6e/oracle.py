"""cuTile port of var_mean_mean_94d5b5ab5e6e: MobileNetV3 BN-train + hard-swish + spatial mean.

Three-kernel design mirroring the Triton reference:
  (1) `_partial_stats_kernel` — per-channel per-chunk sum/sum-of-squares.
  (2) `_finalize_stats_kernel` — reduces partials to mean/invstd; updates running stats.
  (3) `_hardswish_spatial_mean_kernel` — BN affine + bf16 round + hard-swish + spatial mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
VAR_CORRECTION = 1.0000398612827361


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_stats_kernel(
    x_ptr,              # bf16 [N*C, HW] contiguous
    partial_sum_ptr,    # f32 [num_chunks, C]
    partial_sum2_ptr,   # f32 [num_chunks, C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    """Grid: (num_channel_blocks, num_chunks). Each program computes the
    partial sum / sum-of-squares for one channel-block over the n_per_chunk
    n indices in this chunk. We assume 1 chunk per n (n_per_chunk=1) for
    simplicity, so BLOCK_E = HW_PAD, num_chunks = N.
    """
    c_block = ct.bid(0)
    n = ct.bid(1)

    c_start = c_block * C_BLOCK
    hw_arange = ct.arange(HW_PAD, dtype=ct.int32)
    hw_mask_1d = hw_arange < HW

    for c_local in ct.static_iter(range(C_BLOCK)):
        c = c_start + c_local
        c_valid = c < C
        row_idx = n * C + c
        tile = ct.load(
            x_ptr, index=(row_idx, 0), shape=(1, HW_PAD),
            padding_mode=ct.PaddingMode.ZERO,
        )
        tile_1d = ct.astype(ct.reshape(tile, (HW_PAD,)), ct.float32)
        zero_hw = ct.zeros((HW_PAD,), dtype=ct.float32)
        tile_v = ct.where(hw_mask_1d, tile_1d, zero_hw)
        s = ct.sum(tile_v)
        s2 = ct.sum(tile_v * tile_v)
        s_val = ct.where(c_valid, ct.full((1,), s, dtype=ct.float32),
                          ct.zeros((1,), dtype=ct.float32))
        s2_val = ct.where(c_valid, ct.full((1,), s2, dtype=ct.float32),
                           ct.zeros((1,), dtype=ct.float32))
        ct.store(partial_sum_ptr, index=(n, c),
                 tile=ct.reshape(s_val, (1, 1)))
        ct.store(partial_sum2_ptr, index=(n, c),
                 tile=ct.reshape(s2_val, (1, 1)))


@ct.kernel
def _finalize_stats_kernel(
    partial_sum_ptr,     # f32 [N, C]
    partial_sum2_ptr,    # f32 [N, C]
    running_mean_ptr,    # f32 [C]
    running_var_ptr,     # f32 [C]
    saved_mean_ptr,      # f32 [C]
    invstd_ptr,          # f32 [C]
    C: ct.Constant[int],
    N: ct.Constant[int],
    N_PAD: ct.Constant[int],
    E: ct.Constant[int],
    VAR_CORRECTION_: ct.Constant[float],
):
    """One program per channel."""
    c = ct.bid(0)

    # Sum partials along N.
    p_sum = ct.load(partial_sum_ptr, index=(0, c), shape=(N_PAD, 1),
                    padding_mode=ct.PaddingMode.ZERO)
    p_sum2 = ct.load(partial_sum2_ptr, index=(0, c), shape=(N_PAD, 1),
                     padding_mode=ct.PaddingMode.ZERO)
    p_sum_1d = ct.reshape(p_sum, (N_PAD,))
    p_sum2_1d = ct.reshape(p_sum2, (N_PAD,))
    idx = ct.arange(N_PAD, dtype=ct.int32)
    valid = idx < N
    zero_np = ct.zeros((N_PAD,), dtype=ct.float32)
    p_sum_v = ct.where(valid, p_sum_1d, zero_np)
    p_sum2_v = ct.where(valid, p_sum2_1d, zero_np)
    total = ct.sum(p_sum_v)
    total2 = ct.sum(p_sum2_v)

    inv_e = 1.0 / E
    mean = total * inv_e
    ex2 = total2 * inv_e
    var_raw = ex2 - mean * mean
    var = ct.where(var_raw < 0.0, 0.0, var_raw)
    invstd = ct.rsqrt(var + EPS)

    ct.store(saved_mean_ptr, index=(c,),
             tile=ct.reshape(ct.full((1,), mean, dtype=ct.float32), (1,)))
    ct.store(invstd_ptr, index=(c,),
             tile=ct.reshape(ct.full((1,), invstd, dtype=ct.float32), (1,)))

    # Update running stats.
    old_mean = ct.load(running_mean_ptr, index=(c,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(c,), shape=(1,))
    mean_t = ct.reshape(ct.full((1,), mean, dtype=ct.float32), (1,))
    var_t = ct.reshape(ct.full((1,), var, dtype=ct.float32), (1,))
    new_mean = mean_t * 0.1 + old_mean * 0.9
    corrected_var = var_t * VAR_CORRECTION_
    new_var = corrected_var * 0.1 + old_var * 0.9
    ct.store(running_mean_ptr, index=(c,), tile=new_mean)
    ct.store(running_var_ptr, index=(c,), tile=new_var)


@ct.kernel
def _hardswish_spatial_mean_kernel(
    x_ptr,               # bf16 [N*C, HW]
    weight_ptr,          # f32 [C]
    bias_ptr,            # f32 [C]
    saved_mean_ptr,      # f32 [C]
    invstd_ptr,          # f32 [C]
    spatial_mean_ptr,    # bf16 [N*C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
):
    """One program per (n, c) row."""
    row = ct.bid(0)
    c = row - (row // C) * C

    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    mean = ct.load(saved_mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))

    x = ct.load(x_ptr, index=(row, 0), shape=(1, HW_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    x_1d = ct.astype(ct.reshape(x, (HW_PAD,)), ct.float32)
    centered = x_1d - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf, ct.float32)
    # hard-swish: y = x * clamp(x+3, 0, 6) / 6
    plus3 = affine_f + 3.0
    clamped_lo = ct.where(plus3 > 0.0, plus3, 0.0)
    clamped = ct.where(clamped_lo < 6.0, clamped_lo, 6.0)
    hs = (affine_f * clamped) * (1.0 / 6.0)
    hs_bf = ct.astype(hs, ct.bfloat16)
    hs_f = ct.astype(hs_bf, ct.float32)

    cols = ct.arange(HW_PAD, dtype=ct.int32)
    valid = cols < HW
    zero_hw = ct.zeros((HW_PAD,), dtype=ct.float32)
    hs_masked = ct.where(valid, hs_f, zero_hw)
    pooled = ct.sum(hs_masked)
    mean_out = pooled * (1.0 / HW)
    mean_out_bf = ct.astype(mean_out, ct.bfloat16)
    ct.store(spatial_mean_ptr, index=(row,),
             tile=ct.reshape(mean_out_bf, (1,)))


@oracle_impl(hardware="B200", point="d6a317bc")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs
    device = arg0_1.device
    n, c, h, w = arg0_1.shape
    n = int(n); c = int(c); h = int(h); w = int(w)
    hw = h * w
    e = n * hw

    # Reshape to (N*C, HW) — channels-first contiguous view.
    x_flat = arg0_1.permute(0, 1, 2, 3).contiguous().view(n * c, hw)
    HW_PAD = _next_pow2(hw)

    # Allocate partials, saved stats, and output.
    partial_sum = torch.zeros((n, c), device=device, dtype=torch.float32)
    partial_sum2 = torch.zeros((n, c), device=device, dtype=torch.float32)
    saved_mean = torch.empty((c,), device=device, dtype=torch.float32)
    invstd = torch.empty((c,), device=device, dtype=torch.float32)
    spatial_flat = torch.empty((n * c,), device=device, dtype=torch.bfloat16)

    C_BLOCK = 1  # simplest — 1 channel at a time
    N_PAD = _next_pow2(n)

    stream = torch.cuda.current_stream()

    # (1) partial stats
    ct.launch(
        stream, (c // C_BLOCK, n, 1), _partial_stats_kernel,
        (x_flat, partial_sum, partial_sum2, c, hw, HW_PAD, C_BLOCK),
    )
    # (2) finalize stats + running-stats update
    ct.launch(
        stream, (c, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sum2, arg1_1, arg2_1, saved_mean, invstd,
         c, n, N_PAD, e, VAR_CORRECTION),
    )
    # (3) hard-swish + spatial mean
    ct.launch(
        stream, (n * c, 1, 1), _hardswish_spatial_mean_kernel,
        (x_flat, arg3_1, arg4_1, saved_mean, invstd, spatial_flat,
         c, hw, HW_PAD),
    )

    # Assemble outputs to match eager return layout.
    saved_mean_out = saved_mean.view(1, c, 1, 1)
    invstd_out = invstd.view(1, c, 1, 1)
    # spatial_mean shape = shape0 (e.g. [128, 640, 1, 1]) with given strides.
    spatial_mean = torch.empty_strided(
        tuple(int(d) for d in shape0),
        tuple(int(s) for s in shape1),
        device=device, dtype=torch.bfloat16,
    )
    spatial_mean.copy_(spatial_flat.view(n, c, 1, 1))

    return saved_mean_out, invstd_out, spatial_mean, arg1_1, arg2_1
