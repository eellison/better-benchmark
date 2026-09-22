"""cuTile port of var_mean_var_mean_414c9e6634ca: DenseNet stem BN + maxpool + BN.

Four-kernel design (partial-stats/finalize per BN, plus BN affine+relu per BN),
matching Triton's four-kernel structure. Maxpool remains a torch primitive
(matches Triton, which uses `prims._low_memory_max_pool_with_offsets`).

Kernels:
  (1) `_partial_stats_kernel`      — per-channel per-chunk sum/sum-of-squares
                                      for BN1's input.
  (2) `_finalize_and_affine_relu_kernel` — reduce partials, compute mean/invstd,
      update running stats, and produce BN1 affine + ReLU (per-channel).
  (3) `_partial_stats_kernel` (reused) — same for BN2's input (pool output).
  (4) `_finalize_and_affine_relu_kernel` (reused) — same finalize + affine + relu for BN2.

Torch handles maxpool with offsets between the two BN kernel groups (matches the
Triton oracle's dependency on `prims._low_memory_max_pool_with_offsets`).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_stats_kernel(
    x_ptr,             # bf16 [N*C, HW]
    partial_sum_ptr,   # f32 [N, C]
    partial_sum2_ptr,  # f32 [N, C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_TILE: ct.Constant[int],
    CHUNKS_PER_ROW: ct.Constant[int],
):
    """One program per (n, c) — accumulates over HW using static_iter over chunks."""
    n = ct.bid(0)
    c = ct.bid(1)
    row = n * C + c

    acc_s = 0.0
    acc_s2 = 0.0
    for chunk in ct.static_iter(range(CHUNKS_PER_ROW)):
        tile = ct.load(x_ptr, index=(row, chunk), shape=(1, HW_TILE),
                       padding_mode=ct.PaddingMode.ZERO)
        tile_1d = ct.astype(ct.reshape(tile, (HW_TILE,)), ct.float32)
        cols = ct.arange(HW_TILE, dtype=ct.int32) + chunk * HW_TILE
        valid = cols < HW
        zero_hw = ct.zeros((HW_TILE,), dtype=ct.float32)
        tile_v = ct.where(valid, tile_1d, zero_hw)
        acc_s = acc_s + ct.sum(tile_v)
        acc_s2 = acc_s2 + ct.sum(tile_v * tile_v)
    ct.store(partial_sum_ptr, index=(n, c),
             tile=ct.reshape(ct.full((1,), acc_s, dtype=ct.float32), (1, 1)))
    ct.store(partial_sum2_ptr, index=(n, c),
             tile=ct.reshape(ct.full((1,), acc_s2, dtype=ct.float32), (1, 1)))


@ct.kernel
def _finalize_affine_relu_kernel(
    x_ptr,               # bf16 [N*C, HW]
    partial_sum_ptr,     # f32 [N, C]
    partial_sum2_ptr,    # f32 [N, C]
    running_mean_ptr,    # f32 [C]
    running_var_ptr,     # f32 [C]
    weight_ptr,          # f32 [C]
    bias_ptr,            # f32 [C]
    out_ptr,             # bf16 [N*C, HW]
    mean_out_ptr,        # f32 [C]
    invstd_out_ptr,      # f32 [C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    HW_TILE: ct.Constant[int],
    CHUNKS_PER_ROW: ct.Constant[int],
    N: ct.Constant[int],
    N_PAD: ct.Constant[int],
    E: ct.Constant[int],
    VAR_CORRECTION_: ct.Constant[float],
):
    """One program per (n, c). Also updates saved_mean / invstd / running once per c."""
    n = ct.bid(0)
    c = ct.bid(1)
    row = n * C + c

    # Reduce partials for this channel across all n's.
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

    # Only n == 0 writes saved_mean/invstd and running-stats.
    if n == 0:
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

    # BN affine + ReLU (preserve NaN) over the row.
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    for chunk in ct.static_iter(range(CHUNKS_PER_ROW)):
        tile = ct.load(x_ptr, index=(row, chunk), shape=(1, HW_TILE),
                       padding_mode=ct.PaddingMode.ZERO)
        tile_1d = ct.astype(ct.reshape(tile, (HW_TILE,)), ct.float32)
        centered = tile_1d - mean
        normalized = centered * invstd
        affine = normalized * weight + bias
        rounded_bf = ct.astype(affine, ct.bfloat16)
        rounded = ct.astype(rounded_bf, ct.float32)
        is_nan = rounded != rounded
        zero = ct.zeros((HW_TILE,), dtype=ct.float32)
        relu_masked = ct.where(rounded > 0.0, rounded, zero)
        relu = ct.where(is_nan, rounded, relu_masked)
        relu_bf = ct.astype(relu, ct.bfloat16)
        ct.store(out_ptr, index=(row, chunk),
                 tile=ct.reshape(relu_bf, (1, HW_TILE)))


def _run_bn_kernels(x_bf16, weight, bias, running_mean, running_var, var_correction):
    """Run partial+finalize/affine/relu on (N,C,H,W) -> returns bf16 (N,C,H,W),
    plus mean and invstd as f32 [1,C,1,1]."""
    N, C, H, W = x_bf16.shape
    device = x_bf16.device
    hw = H * W
    e = N * hw
    x_flat = x_bf16.permute(0, 1, 2, 3).contiguous().view(N * C, hw)

    # Chunk over HW.
    HW_TILE = min(_next_pow2(hw), 1024)
    chunks_per_row = (hw + HW_TILE - 1) // HW_TILE

    partial_sum = torch.zeros((N, C), device=device, dtype=torch.float32)
    partial_sum2 = torch.zeros((N, C), device=device, dtype=torch.float32)
    out_flat = torch.empty((N * C, hw), device=device, dtype=torch.bfloat16)
    mean_out = torch.empty((C,), device=device, dtype=torch.float32)
    invstd_out = torch.empty((C,), device=device, dtype=torch.float32)
    N_PAD = _next_pow2(N)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, C, 1), _partial_stats_kernel,
        (x_flat, partial_sum, partial_sum2, C, hw, HW_TILE, chunks_per_row),
    )
    ct.launch(
        stream, (N, C, 1), _finalize_affine_relu_kernel,
        (x_flat, partial_sum, partial_sum2, running_mean, running_var,
         weight, bias, out_flat, mean_out, invstd_out,
         C, hw, HW_TILE, chunks_per_row, N, N_PAD, e, var_correction),
    )
    out_bf16 = out_flat.view(N, C, H, W)
    return out_bf16, mean_out.view(1, C, 1, 1), invstd_out.view(1, C, 1, 1)


@oracle_impl(hardware="B200", point="45fdeec8")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     _shape_param_0, _shape_param_1) = inputs

    # BN1 + ReLU via cuTile kernels
    relu1, mean1, invstd1 = _run_bn_kernels(
        arg0_1, arg3_1, arg4_1, arg1_1, arg2_1, 1.0000199302441455,
    )

    # Maxpool (torch prims — matches Triton dependency on prims maxpool w/ offsets)
    pool_out, pool_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu1, [3, 3], [2, 2], [1, 1], [1, 1], False)

    # BN2 + ReLU via cuTile kernels
    relu2, mean2_4d, invstd2_4d = _run_bn_kernels(
        pool_out, arg7_1, arg8_1, arg5_1, arg6_1, 1.0000797257434426,
    )

    c = int(arg0_1.shape[1])
    squeeze_3 = invstd2_4d.view(c)
    unsqueeze_10 = mean2_4d
    return (mean1, invstd1, pool_out, pool_offsets, squeeze_3, relu2,
            unsqueeze_10, arg1_1, arg2_1, arg5_1, arg6_1)
