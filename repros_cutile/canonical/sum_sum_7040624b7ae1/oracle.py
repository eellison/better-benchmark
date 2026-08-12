"""cuTile port of sum_sum_7040624b7ae1: GhostNet masked BN-backward.

Mirrors Triton's 3-kernel split-K structure:
- _partial_reduce_kernel: per-channel partial sums via `ct.sum(..., axis=0)`.
- _finalize_kernel: reduce partials per channel + compute coefficients.
- _epilogue_kernel: full elementwise BN-backward output (channels-last).

All in-kernel reductions use `ct.sum`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 184
C = 92
H = 14
W = 14
HW = H * W
K_TOTAL = N * HW
TOTAL = N * C * HW
SCALE = 9.964923469387754e-06


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    arg0_ptr,      # bf16 [N*C_IN*HW] flat (channels-last)
    arg1_ptr,      # bf16 [N*C*HW] flat (channels-last)
    arg2_ptr,      # bf16 [N*C*HW] flat (channels-last)
    fill_ptr,      # bf16 [1]
    arg4_ptr,      # bf16 [N*C*HW] flat (channels-last)
    mean_ptr,      # f32 [C]
    partial_sum_ptr,   # f32 [num_tiles, C]
    partial_dot_ptr,   # f32 [num_tiles, C]
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    k = tile * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    active = k < K_TOTAL
    n = k // HW
    hw = k - n * HW
    # channels-last offsets: (n * C * HW) + hw * C + c
    offset_c = n * (C * HW) + hw * C + c
    # arg0 is 184-channel input, sliced to first 92 channels
    offset_cin = n * (C_IN * HW) + hw * C_IN + c

    lhs_bf = ct.gather(arg0_ptr, offset_cin)
    rhs_bf = ct.gather(arg1_ptr, offset_c)
    mask_bf = ct.gather(arg2_ptr, offset_c)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)

    added_f = ct.astype(lhs_bf, ct.float32) + ct.astype(rhs_bf, ct.float32)
    added_bf = ct.astype(added_f, ct.bfloat16)
    added_r = ct.astype(added_bf, ct.float32)
    mask_f = ct.astype(mask_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_K,), dtype=ct.float32)
    source = ct.where(mask_f <= zero_f,
                      ct.broadcast_to(fill_f, (BLOCK_K,)), added_r)
    source = ct.where(active, source, 0.0)

    arg4_bf = ct.gather(arg4_ptr, offset_c)
    arg4_f = ct.astype(arg4_bf, ct.float32)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean_bc = ct.broadcast_to(mean_scalar, (BLOCK_K,))
    centered = ct.where(active, arg4_f - mean_bc, 0.0)

    prod = source * centered
    sum_v = ct.sum(source)
    dot_v = ct.sum(prod)

    partial_off = tile * C + c
    ct.scatter(partial_sum_ptr, ct.broadcast_to(partial_off, (1,)),
               ct.reshape(sum_v, (1,)))
    ct.scatter(partial_dot_ptr, ct.broadcast_to(partial_off, (1,)),
               ct.reshape(dot_v, (1,)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [num_tiles, C]
    partial_dot_ptr,   # f32 [num_tiles, C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    scaled_dot_out_ptr, # f32 [C]
    mean_term_ptr,     # f32 [C]
    coeff_ptr,         # f32 [C]
    output_scale_ptr,  # f32 [C]
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    tiles = ct.arange(BLOCK_TILES, dtype=ct.int32)
    active = tiles < NUM_TILES
    offsets = c + C * tiles
    zero_i = ct.zeros((BLOCK_TILES,), dtype=ct.int32)
    safe_off = ct.where(active, offsets, zero_i)

    sum_vals = ct.gather(partial_sum_ptr, safe_off)
    dot_vals = ct.gather(partial_dot_ptr, safe_off)
    sum_vals = ct.where(active, sum_vals, 0.0)
    dot_vals = ct.where(active, dot_vals, 0.0)
    sum_v = ct.sum(sum_vals)
    dot_v = ct.sum(dot_vals)

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_scalar = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, ())
    weight = ct.reshape(weight_scalar, ())
    invstd_sq = invstd * invstd
    scaled_dot = dot_v * invstd

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_v, (1,)))
    ct.store(scaled_dot_out_ptr, index=(c,), tile=ct.reshape(scaled_dot, (1,)))
    ct.store(mean_term_ptr, index=(c,), tile=ct.reshape(sum_v * SCALE, (1,)))
    ct.store(coeff_ptr, index=(c,), tile=ct.reshape(dot_v * SCALE * invstd_sq, (1,)))
    ct.store(output_scale_ptr, index=(c,), tile=ct.reshape(invstd * weight, (1,)))


@ct.kernel
def _epilogue_kernel(
    arg0_ptr,      # bf16 [N*C_IN*HW]
    arg1_ptr,      # bf16 [N*C*HW]
    arg2_ptr,      # bf16 [N*C*HW]
    fill_ptr,      # bf16 [1]
    arg4_ptr,      # bf16 [N*C*HW]
    mean_ptr,      # f32 [C]
    mean_term_ptr, # f32 [C]
    coeff_ptr,     # f32 [C]
    output_scale_ptr, # f32 [C]
    out_ptr,       # bf16 [N*C*HW]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    linear = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = linear < TOTAL
    c = linear - (linear // C) * C
    k = linear // C
    n = k // HW
    hw = k - n * HW

    # arg0 is 184-channel input; arg1..arg4 are 92-channel.
    # arg1/arg2/arg4 use same channels-last packing as arg0 slice_1.
    # linear index in the output (also 92-ch channels-last) is `linear`.
    offset_c = n * (C * HW) + hw * C + c
    offset_cin = n * (C_IN * HW) + hw * C_IN + c

    lhs_bf = ct.gather(arg0_ptr, offset_cin)
    rhs_bf = ct.gather(arg1_ptr, offset_c)
    mask_bf = ct.gather(arg2_ptr, offset_c)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)
    added_f = ct.astype(lhs_bf, ct.float32) + ct.astype(rhs_bf, ct.float32)
    added_bf = ct.astype(added_f, ct.bfloat16)
    added_r = ct.astype(added_bf, ct.float32)
    mask_f = ct.astype(mask_bf, ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    source = ct.where(mask_f <= zero_f,
                      ct.broadcast_to(fill_f, (BLOCK,)), added_r)

    arg4_bf = ct.gather(arg4_ptr, offset_c)
    arg4_f = ct.astype(arg4_bf, ct.float32)
    mean_v = ct.gather(mean_ptr, c)
    centered = arg4_f - mean_v
    coeff = ct.gather(coeff_ptr, c)
    mean_term = ct.gather(mean_term_ptr, c)
    output_scale = ct.gather(output_scale_ptr, c)
    correction = centered * coeff
    corrected = source - correction
    corrected = corrected - mean_term
    out_f = corrected * output_scale
    # NOTE: Triton stores f32 to bf16 output pointer; PTX will convert on store.
    # We match Triton by casting explicitly to bf16.
    out_bf = ct.astype(out_f, ct.bfloat16)
    ct.scatter(out_ptr, linear, out_bf, mask=active)


@oracle_impl(
    hardware="B200",
    point="26501e14",
    BLOCK_K=128,
    EPILOGUE_BLOCK=256,
)
def oracle_forward(inputs, *, BLOCK_K: int, EPILOGUE_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device
    num_tiles = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    block_tiles = _next_power_of_2(num_tiles)

    partial_sum = torch.empty((num_tiles * C,), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles * C,), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )

    def _flat_cl(t):
        # For channels-last tensor (N,C,H,W) with stride (C*HW,1,W*C,C):
        # the storage is arranged as [n=0][hw=0][c=0..C-1][hw=1][c=0..C-1]...
        # Storage size = N*C*HW
        return torch.as_strided(t, (t.numel(),), (1,), 0)

    arg0_flat = _flat_cl(arg0_1)
    arg1_flat = _flat_cl(arg1_1)
    arg2_flat = _flat_cl(arg2_1)
    arg4_flat = _flat_cl(arg4_1)
    mean_1d = arg5_1.view(C)
    invstd_1d = arg6_1.view(C)
    weight_1d = arg7_1.view(C)
    fill_1d = arg3_1.view(1)
    out_flat = _flat_cl(out)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, num_tiles, 1),
        _partial_reduce_kernel,
        (arg0_flat, arg1_flat, arg2_flat, fill_1d, arg4_flat,
         mean_1d, partial_sum, partial_dot, BLOCK_K),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, scaled_dot_out, mean_term, coeff, output_scale,
         num_tiles, block_tiles),
    )
    ct.launch(
        stream,
        ((TOTAL + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK, 1, 1),
        _epilogue_kernel,
        (arg0_flat, arg1_flat, arg2_flat, fill_1d, arg4_flat,
         mean_1d, mean_term, coeff, output_scale, out_flat,
         EPILOGUE_BLOCK),
    )
    return sum_out, scaled_dot_out, out
