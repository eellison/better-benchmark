"""cuTile port of var_mean_mean_6a2d0c92a5ed: MobileNetV3 BN + hard-swish + spatial mean.

Three-kernel design mirroring the Triton reference:
  (1) `_partial_stats_kernel` — per-channel partial sum / sum-of-squares.
  (2) `_finalize_stats_kernel` — reduces partials to mean/invstd; updates running stats.
  (3) `_hardswish_spatial_mean_kernel` — BN affine + bf16 round + hard-swish + spatial mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_stats_kernel(
    x_ptr,              # bf16 [N*C, HW] (channels-first flat)
    partial_sum_ptr,    # f32 [N, C]
    partial_sum2_ptr,   # f32 [N, C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
):
    """One program per (n, c)."""
    n = ct.bid(0)
    c = ct.bid(1)
    row = n * C + c

    hw_arange = ct.arange(HW_PAD, dtype=ct.int32)
    hw_mask = hw_arange < HW

    tile = ct.load(x_ptr, index=(row, 0), shape=(1, HW_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_1d = ct.astype(ct.reshape(tile, (HW_PAD,)), ct.float32)
    zero_hw = ct.zeros((HW_PAD,), dtype=ct.float32)
    tile_v = ct.where(hw_mask, tile_1d, zero_hw)
    s = ct.sum(tile_v)
    s2 = ct.sum(tile_v * tile_v)
    ct.store(partial_sum_ptr, index=(n, c),
             tile=ct.reshape(ct.full((1,), s, dtype=ct.float32), (1, 1)))
    ct.store(partial_sum2_ptr, index=(n, c),
             tile=ct.reshape(ct.full((1,), s2, dtype=ct.float32), (1, 1)))


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
):
    """One program per channel."""
    c = ct.bid(0)
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

    old_mean = ct.load(running_mean_ptr, index=(c,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(c,), shape=(1,))
    mean_t = ct.reshape(ct.full((1,), mean, dtype=ct.float32), (1,))
    var_t = ct.reshape(ct.full((1,), var, dtype=ct.float32), (1,))
    new_mean = mean_t * MOMENTUM + old_mean * (1.0 - MOMENTUM)
    corrected = var_t * RUNNING_VAR_CORRECTION
    new_var = corrected * MOMENTUM + old_var * (1.0 - MOMENTUM)
    ct.store(running_mean_ptr, index=(c,), tile=new_mean)
    ct.store(running_var_ptr, index=(c,), tile=new_var)


@ct.kernel
def _hardswish_spatial_mean_kernel(
    x_ptr,               # bf16 [N*C, HW]
    saved_mean_ptr,      # f32 [C]
    invstd_ptr,          # f32 [C]
    weight_ptr,          # f32 [C]
    bias_ptr,            # f32 [C]
    act_ptr,             # bf16 [N*C, HW]
    spatial_mean_ptr,    # bf16 [N*C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
):
    """One program per (n, c) row."""
    row = ct.bid(0)
    c = row - (row // C) * C

    mean = ct.load(saved_mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))

    x = ct.load(x_ptr, index=(row, 0), shape=(1, HW_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    x_1d = ct.astype(ct.reshape(x, (HW_PAD,)), ct.float32)
    centered = x_1d - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    rounded_bf = ct.astype(affine, ct.bfloat16)
    rounded = ct.astype(rounded_bf, ct.float32)
    plus3 = rounded + 3.0
    clamped_lo = ct.where(plus3 > 0.0, plus3, 0.0)
    clamped = ct.where(clamped_lo < 6.0, clamped_lo, 6.0)
    hs = (rounded * clamped) * (1.0 / 6.0)
    hs_bf = ct.astype(hs, ct.bfloat16)

    ct.store(act_ptr, index=(row, 0),
             tile=ct.reshape(hs_bf, (1, HW_PAD)))

    cols = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = cols < HW
    zero_hw = ct.zeros((HW_PAD,), dtype=ct.float32)
    hs_f = ct.astype(hs_bf, ct.float32)
    hs_masked = ct.where(hw_valid, hs_f, zero_hw)
    reduced = ct.sum(hs_masked)
    mean_out = reduced * (1.0 / HW)
    mean_out_bf = ct.astype(mean_out, ct.bfloat16)
    ct.store(spatial_mean_ptr, index=(row,),
             tile=ct.reshape(mean_out_bf, (1,)))


@oracle_impl(hardware="B200", point="d6a317bc", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=64, ROW_BLOCK=32)
@oracle_impl(hardware="B200", point="c5a5573b", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=64, ROW_BLOCK=32)
@oracle_impl(hardware="B200", point="055f7aad", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=256, ROW_BLOCK=8)
@oracle_impl(hardware="B200", point="bb4cfa64", BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=256, ROW_BLOCK=8)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, BLOCK_HW: int, ROW_BLOCK: int):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
    device = x.device

    # NCHW contiguous, then flatten to (N*C, HW).
    x_flat = x.permute(0, 1, 2, 3).contiguous().view(n * c, hw)
    HW_PAD = _next_pow2(hw)

    partial_sum = torch.zeros((n, c), device=device, dtype=torch.float32)
    partial_sum2 = torch.zeros((n, c), device=device, dtype=torch.float32)
    saved_mean = torch.empty((c,), device=device, dtype=torch.float32)
    invstd = torch.empty((c,), device=device, dtype=torch.float32)
    act_flat = torch.empty((n * c, hw), device=device, dtype=torch.bfloat16)
    spatial_flat = torch.empty((n * c,), device=device, dtype=torch.bfloat16)

    N_PAD = _next_pow2(n)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream, (n, c, 1), _partial_stats_kernel,
        (x_flat, partial_sum, partial_sum2, c, hw, HW_PAD),
    )
    ct.launch(
        stream, (c, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sum2, running_mean, running_var,
         saved_mean, invstd, c, n, N_PAD, e),
    )
    ct.launch(
        stream, (n * c, 1, 1), _hardswish_spatial_mean_kernel,
        (x_flat, saved_mean, invstd, weight, bias, act_flat, spatial_flat,
         c, hw, HW_PAD),
    )

    # Assemble outputs.
    saved_mean_out = saved_mean.view(1, c, 1, 1)
    invstd_out = invstd.view(1, c, 1, 1)
    activation = torch.empty_strided(tuple(x.shape), tuple(x.stride()),
                                       device=device, dtype=torch.bfloat16)
    activation.copy_(act_flat.view(n, c, h, w))
    spatial_mean = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1),
                                         device=device, dtype=torch.bfloat16)
    spatial_mean.copy_(spatial_flat.view(n, c, 1, 1))
    return saved_mean_out, invstd_out, activation, spatial_mean, running_mean, running_var
