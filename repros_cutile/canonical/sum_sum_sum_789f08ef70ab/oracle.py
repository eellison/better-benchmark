"""cuTile port of sum_sum_sum_789f08ef70ab: ShuffleNet channel-shuffle +
dual BatchNorm-backward tuple.

Strategy:
  - Torch: precompute the channel-shuffle output view_1 with the eager
    layout (cat -> view -> permute -> clone -> view). Extract slice_2
    (even channels) and slice_3 (odd channels) as views.
  - cuTile partial kernel: for each (batch, hw_block, channel_block) chunk
    compute the two "where(le, arg7, shuffled)" f32 producers and emit
    per-channel partial sums (sum_a, dot_a, sum_b, dot_b) into a
    partials tensor.
  - cuTile finalize kernel: reduce partials -> sum_1, mul_10, sum_3, mul_21
    + a 6xC stats tile used by the epilogue.
  - cuTile epilogue kernel: computes the two BN-backward bf16 outputs
    from cached stats.

Inputs come in NCHW logical shape with channels-last strides
(11368, 1, 1624, 232) which is memory-format NHWC. We permute to
[N, HW, C] contiguous for cuTile access.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 232
H = 7
W = 7
HW = H * W
SPATIAL_TOTAL = N * HW  # 6272
SCALE = 0.00015943877551020407


@ct.kernel
def _partial_kernel(
    slice3_ptr,     # bf16 [N, HW, C]  (odd-shuffle, channels-last flattened)
    slice2_ptr,     # bf16 [N, HW, C]  (even-shuffle)
    xa_ptr,         # bf16 [N, HW, C]  (arg2_1)
    mean_a_ptr,     # f32 [C]  (arg3_1 squeezed)
    invstd_a_ptr,   # f32 [C]  (arg4_1 squeezed)
    weight_a_ptr,   # f32 [C]  (arg5_1)
    bias_a_ptr,     # f32 [C]  (arg6_1)
    zero_ptr,       # bf16 [1]  (arg7_1)
    xb_ptr,         # bf16 [N, HW, C]  (arg8_1)
    mean_b_ptr,     # f32 [C]  (arg9_1)
    invstd_b_ptr,   # f32 [C]  (arg10_1)
    weight_b_ptr,   # f32 [C]  (arg11_1)
    bias_b_ptr,     # f32 [C]  (arg12_1)
    partials_ptr,   # f32 [4, C, NUM_TILES]  flat
    NUM_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile_id = ct.bid(0)  # spatial tile index (rows split)
    c_block = ct.bid(1)  # channel block index

    c_offsets = ct.arange(BLOCK_C, dtype=ct.int64) + (c_block * BLOCK_C)
    c_mask = c_offsets < C

    weight_a = ct.load(weight_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    weight_b = ct.load(weight_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    bias_a = ct.load(bias_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias_b = ct.load(bias_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    mean_a = ct.load(mean_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    invstd_a = ct.load(invstd_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    mean_b = ct.load(mean_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    invstd_b = ct.load(invstd_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)

    zero_bf_tile = ct.load(zero_ptr, index=(0,), shape=(1,))
    zero_f_1 = ct.astype(zero_bf_tile, ct.float32)
    zero_f_3d = ct.reshape(zero_f_1, (1, 1, 1))
    zero_broad = ct.broadcast_to(zero_f_3d, (BLOCK_N, BLOCK_HW, BLOCK_C))

    acc_sum_a = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_dot_a = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_sum_b = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_dot_b = ct.zeros((BLOCK_C,), dtype=ct.float32)

    # Each partial-kernel launch iterates over one spatial tile at a time.
    # tile_id ranges over cdiv(N * HW, BLOCK_N*BLOCK_HW).
    # We flatten (n, hw) into one linear index for simplicity: iterate one
    # micro-tile from the [N, HW, C] view. We pick BLOCK_N=1 and BLOCK_HW
    # divides HW (padded). We iterate: n = tile_id // n_hw_tiles, hw_block
    # = tile_id % n_hw_tiles.
    # Use direct 3D indexing with (n, hw_block, c_block).
    # Load slice3, slice2, xa, xb, tiles of shape (BLOCK_N, BLOCK_HW, BLOCK_C).
    n_hw_tiles = ct.cdiv(HW, BLOCK_HW)
    n_idx = tile_id // n_hw_tiles
    hw_idx = tile_id - n_idx * n_hw_tiles

    # Load 3D tiles.
    slice3 = ct.load(
        slice3_ptr, index=(n_idx, hw_idx, c_block),
        shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    slice2 = ct.load(
        slice2_ptr, index=(n_idx, hw_idx, c_block),
        shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    xa = ct.load(
        xa_ptr, index=(n_idx, hw_idx, c_block),
        shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    xb = ct.load(
        xb_ptr, index=(n_idx, hw_idx, c_block),
        shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # Build hw_mask & c_mask
    hw_offsets = ct.arange(BLOCK_HW, dtype=ct.int64) + (hw_idx * BLOCK_HW)
    hw_valid = hw_offsets < HW
    n_offsets = ct.arange(BLOCK_N, dtype=ct.int64) + (n_idx * BLOCK_N)
    n_valid = n_offsets < N
    mask_n = ct.reshape(n_valid, (BLOCK_N, 1, 1))
    mask_hw = ct.reshape(hw_valid, (1, BLOCK_HW, 1))
    mask_c = ct.reshape(c_mask, (1, 1, BLOCK_C))
    mask3d = mask_n & mask_hw & mask_c

    # Broadcast per-channel stats to (BLOCK_N, BLOCK_HW, BLOCK_C)
    weight_a_3 = ct.reshape(weight_a, (1, 1, BLOCK_C))
    weight_b_3 = ct.reshape(weight_b, (1, 1, BLOCK_C))
    bias_a_3 = ct.reshape(bias_a, (1, 1, BLOCK_C))
    bias_b_3 = ct.reshape(bias_b, (1, 1, BLOCK_C))
    mean_a_3 = ct.reshape(mean_a, (1, 1, BLOCK_C))
    invstd_a_3 = ct.reshape(invstd_a, (1, 1, BLOCK_C))
    mean_b_3 = ct.reshape(mean_b, (1, 1, BLOCK_C))
    invstd_b_3 = ct.reshape(invstd_b, (1, 1, BLOCK_C))

    xa_f = ct.astype(xa, ct.float32)
    centered_a = xa_f - mean_a_3
    affine_a = centered_a * invstd_a_3 * weight_a_3 + bias_a_3
    affine_a_bf = ct.astype(affine_a, ct.bfloat16)
    zero_bf_3 = ct.full((BLOCK_N, BLOCK_HW, BLOCK_C), 0.0, dtype=ct.bfloat16)
    le_a = affine_a_bf <= zero_bf_3
    slice3_f = ct.astype(slice3, ct.float32)
    where_a = ct.where(le_a, zero_broad, slice3_f)

    xb_f = ct.astype(xb, ct.float32)
    centered_b = xb_f - mean_b_3
    affine_b = centered_b * invstd_b_3 * weight_b_3 + bias_b_3
    affine_b_bf = ct.astype(affine_b, ct.bfloat16)
    le_b = affine_b_bf <= zero_bf_3
    slice2_f = ct.astype(slice2, ct.float32)
    where_b = ct.where(le_b, zero_broad, slice2_f)

    # Sum over batch(N) and spatial(HW), keep channel(C).
    zero_3d = ct.full((BLOCK_N, BLOCK_HW, BLOCK_C), 0.0, dtype=ct.float32)
    wa_masked = ct.where(mask3d, where_a, zero_3d)
    wb_masked = ct.where(mask3d, where_b, zero_3d)
    ca_masked = ct.where(mask3d, centered_a, zero_3d)
    cb_masked = ct.where(mask3d, centered_b, zero_3d)

    partial_sum_a = ct.sum(wa_masked, axis=1)
    partial_sum_a = ct.sum(partial_sum_a, axis=0)  # (BLOCK_C,)
    partial_dot_a = ct.sum(wa_masked * ca_masked, axis=1)
    partial_dot_a = ct.sum(partial_dot_a, axis=0)
    partial_sum_b = ct.sum(wb_masked, axis=1)
    partial_sum_b = ct.sum(partial_sum_b, axis=0)
    partial_dot_b = ct.sum(wb_masked * cb_masked, axis=1)
    partial_dot_b = ct.sum(partial_dot_b, axis=0)

    # Store partials at [4, C, NUM_TILES] flat layout: index (which*C + c)*NT + tile_id.
    plane = C * NUM_TILES
    base_offsets = c_offsets * NUM_TILES + tile_id
    big1d = ct.full((BLOCK_C,), 10**12, dtype=ct.int64)
    off_a = ct.where(c_mask, base_offsets, big1d)
    off_b = ct.where(c_mask, base_offsets + plane, big1d)
    off_c = ct.where(c_mask, base_offsets + 2 * plane, big1d)
    off_d = ct.where(c_mask, base_offsets + 3 * plane, big1d)
    ct.scatter(partials_ptr, off_a, partial_sum_a)
    ct.scatter(partials_ptr, off_b, partial_dot_a)
    ct.scatter(partials_ptr, off_c, partial_sum_b)
    ct.scatter(partials_ptr, off_d, partial_dot_b)


@ct.kernel
def _finalize_kernel(
    partials_ptr,
    invstd_a_ptr,
    weight_a_ptr,
    invstd_b_ptr,
    weight_b_ptr,
    sum_a_ptr,      # f32 [C]
    vec_a_ptr,      # f32 [C]  = sum_2 * squeeze_1
    sum_b_ptr,      # f32 [C]
    vec_b_ptr,      # f32 [C]
    stats_ptr,      # f32 [6, C] flat
    NUM_TILES: ct.Constant[int],
    BLOCK_T: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    c_offsets = ct.arange(BLOCK_C, dtype=ct.int64) + (c_block * BLOCK_C)
    c_mask = c_offsets < C
    tile_offsets = ct.arange(BLOCK_T, dtype=ct.int64)
    t_mask = tile_offsets < NUM_TILES

    plane = C * NUM_TILES

    # Load partials as (BLOCK_T, BLOCK_C) — index = tile + c * NUM_TILES
    # We need to compute a broadcasted index tile.
    # Use expand_dims: tiles_2d (T, 1), cols_2d (1, C).
    tiles_2d = ct.reshape(tile_offsets, (BLOCK_T, 1))
    cols_2d = ct.reshape(c_offsets, (1, BLOCK_C))
    offs_2d = cols_2d * NUM_TILES + tiles_2d
    tmask_2d = ct.reshape(t_mask, (BLOCK_T, 1))
    cmask_2d = ct.reshape(c_mask, (1, BLOCK_C))
    mask2d = tmask_2d & cmask_2d

    big_2d = ct.full((BLOCK_T, BLOCK_C), 10**12, dtype=ct.int64)
    sa_idx = ct.where(mask2d, offs_2d, big_2d)
    da_idx = ct.where(mask2d, offs_2d + plane, big_2d)
    sb_idx = ct.where(mask2d, offs_2d + 2 * plane, big_2d)
    db_idx = ct.where(mask2d, offs_2d + 3 * plane, big_2d)

    sa = ct.gather(partials_ptr, sa_idx)
    da = ct.gather(partials_ptr, da_idx)
    sb = ct.gather(partials_ptr, sb_idx)
    db = ct.gather(partials_ptr, db_idx)
    zero_2d = ct.zeros((BLOCK_T, BLOCK_C), dtype=ct.float32)
    sa = ct.where(mask2d, sa, zero_2d)
    da = ct.where(mask2d, da, zero_2d)
    sb = ct.where(mask2d, sb, zero_2d)
    db = ct.where(mask2d, db, zero_2d)

    sum_a = ct.sum(sa, axis=0)  # (BLOCK_C,)
    dot_a = ct.sum(da, axis=0)
    sum_b = ct.sum(sb, axis=0)
    dot_b = ct.sum(db, axis=0)

    invstd_a = ct.load(invstd_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    weight_a = ct.load(weight_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    invstd_b = ct.load(invstd_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    weight_b = ct.load(weight_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)

    # Prepare masked scatter indices.
    big_1d = ct.full((BLOCK_C,), 10**9, dtype=ct.int64)
    c_safe = ct.where(c_mask, c_offsets, big_1d)
    ct.scatter(sum_a_ptr, c_safe, sum_a)
    ct.scatter(vec_a_ptr, c_safe, dot_a * invstd_a)
    ct.scatter(sum_b_ptr, c_safe, sum_b)
    ct.scatter(vec_b_ptr, c_safe, dot_b * invstd_b)

    # Stats layout [6, C] flat: mean_term_a, coeff_a, scale_a, mean_term_b, coeff_b, scale_b.
    off_stats0 = ct.where(c_mask, c_offsets, big_1d)
    off_stats1 = ct.where(c_mask, c_offsets + C, big_1d)
    off_stats2 = ct.where(c_mask, c_offsets + 2 * C, big_1d)
    off_stats3 = ct.where(c_mask, c_offsets + 3 * C, big_1d)
    off_stats4 = ct.where(c_mask, c_offsets + 4 * C, big_1d)
    off_stats5 = ct.where(c_mask, c_offsets + 5 * C, big_1d)
    ct.scatter(stats_ptr, off_stats0, sum_a * SCALE)
    ct.scatter(stats_ptr, off_stats1, (dot_a * SCALE) * (invstd_a * invstd_a))
    ct.scatter(stats_ptr, off_stats2, invstd_a * weight_a)
    ct.scatter(stats_ptr, off_stats3, sum_b * SCALE)
    ct.scatter(stats_ptr, off_stats4, (dot_b * SCALE) * (invstd_b * invstd_b))
    ct.scatter(stats_ptr, off_stats5, invstd_b * weight_b)


@ct.kernel
def _epilogue_kernel(
    slice3_ptr, slice2_ptr,
    xa_ptr, mean_a_ptr, invstd_a_ptr, weight_a_ptr, bias_a_ptr,
    zero_ptr,
    xb_ptr, mean_b_ptr, invstd_b_ptr, weight_b_ptr, bias_b_ptr,
    stats_ptr,      # f32 [6, C] flat
    out_a_ptr,      # bf16 [N, HW, C]
    out_b_ptr,
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n_idx = ct.bid(0)
    hw_idx = ct.bid(1)
    c_block = ct.bid(2)

    c_offsets = ct.arange(BLOCK_C, dtype=ct.int64) + (c_block * BLOCK_C)
    c_mask = c_offsets < C
    hw_offsets = ct.arange(BLOCK_HW, dtype=ct.int64) + (hw_idx * BLOCK_HW)
    hw_valid = hw_offsets < HW
    n_offsets = ct.arange(BLOCK_N, dtype=ct.int64) + (n_idx * BLOCK_N)
    n_valid = n_offsets < N
    mask_n = ct.reshape(n_valid, (BLOCK_N, 1, 1))
    mask_hw = ct.reshape(hw_valid, (1, BLOCK_HW, 1))
    mask_c = ct.reshape(c_mask, (1, 1, BLOCK_C))
    mask3d = mask_n & mask_hw & mask_c

    slice3 = ct.load(slice3_ptr, index=(n_idx, hw_idx, c_block),
                     shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    slice2 = ct.load(slice2_ptr, index=(n_idx, hw_idx, c_block),
                     shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    xa = ct.load(xa_ptr, index=(n_idx, hw_idx, c_block),
                 shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    xb = ct.load(xb_ptr, index=(n_idx, hw_idx, c_block),
                 shape=(BLOCK_N, BLOCK_HW, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)

    weight_a = ct.load(weight_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    weight_b = ct.load(weight_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    bias_a = ct.load(bias_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias_b = ct.load(bias_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    mean_a = ct.load(mean_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    invstd_a = ct.load(invstd_a_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    mean_b = ct.load(mean_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    invstd_b = ct.load(invstd_b_ptr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)

    zero_bf_tile = ct.load(zero_ptr, index=(0,), shape=(1,))
    zero_f_1 = ct.astype(zero_bf_tile, ct.float32)
    zero_f_3d = ct.reshape(zero_f_1, (1, 1, 1))
    zero_broad = ct.broadcast_to(zero_f_3d, (BLOCK_N, BLOCK_HW, BLOCK_C))

    weight_a_3 = ct.reshape(weight_a, (1, 1, BLOCK_C))
    weight_b_3 = ct.reshape(weight_b, (1, 1, BLOCK_C))
    bias_a_3 = ct.reshape(bias_a, (1, 1, BLOCK_C))
    bias_b_3 = ct.reshape(bias_b, (1, 1, BLOCK_C))
    mean_a_3 = ct.reshape(mean_a, (1, 1, BLOCK_C))
    invstd_a_3 = ct.reshape(invstd_a, (1, 1, BLOCK_C))
    mean_b_3 = ct.reshape(mean_b, (1, 1, BLOCK_C))
    invstd_b_3 = ct.reshape(invstd_b, (1, 1, BLOCK_C))

    xa_f = ct.astype(xa, ct.float32)
    centered_a = xa_f - mean_a_3
    affine_a = centered_a * invstd_a_3 * weight_a_3 + bias_a_3
    affine_a_bf = ct.astype(affine_a, ct.bfloat16)
    zero_bf_3 = ct.full((BLOCK_N, BLOCK_HW, BLOCK_C), 0.0, dtype=ct.bfloat16)
    le_a = affine_a_bf <= zero_bf_3
    slice3_f = ct.astype(slice3, ct.float32)
    where_a = ct.where(le_a, zero_broad, slice3_f)

    xb_f = ct.astype(xb, ct.float32)
    centered_b = xb_f - mean_b_3
    affine_b = centered_b * invstd_b_3 * weight_b_3 + bias_b_3
    affine_b_bf = ct.astype(affine_b, ct.bfloat16)
    le_b = affine_b_bf <= zero_bf_3
    slice2_f = ct.astype(slice2, ct.float32)
    where_b = ct.where(le_b, zero_broad, slice2_f)

    # Load stats
    big_1d = ct.full((BLOCK_C,), 10**9, dtype=ct.int64)
    c_safe = ct.where(c_mask, c_offsets, big_1d)
    zeros_c = ct.zeros((BLOCK_C,), dtype=ct.float32)
    mean_term_a = ct.where(c_mask, ct.gather(stats_ptr, c_safe), zeros_c)
    coeff_a = ct.where(c_mask, ct.gather(stats_ptr, c_safe + C), zeros_c)
    scale_a = ct.where(c_mask, ct.gather(stats_ptr, c_safe + 2 * C), zeros_c)
    mean_term_b = ct.where(c_mask, ct.gather(stats_ptr, c_safe + 3 * C), zeros_c)
    coeff_b = ct.where(c_mask, ct.gather(stats_ptr, c_safe + 4 * C), zeros_c)
    scale_b = ct.where(c_mask, ct.gather(stats_ptr, c_safe + 5 * C), zeros_c)

    mean_term_a_3 = ct.reshape(mean_term_a, (1, 1, BLOCK_C))
    coeff_a_3 = ct.reshape(coeff_a, (1, 1, BLOCK_C))
    scale_a_3 = ct.reshape(scale_a, (1, 1, BLOCK_C))
    mean_term_b_3 = ct.reshape(mean_term_b, (1, 1, BLOCK_C))
    coeff_b_3 = ct.reshape(coeff_b, (1, 1, BLOCK_C))
    scale_b_3 = ct.reshape(scale_b, (1, 1, BLOCK_C))

    grad_a = ((where_a - centered_a * coeff_a_3) - mean_term_a_3) * scale_a_3
    grad_b = ((where_b - centered_b * coeff_b_3) - mean_term_b_3) * scale_b_3

    ct.store(out_a_ptr, index=(n_idx, hw_idx, c_block),
             tile=ct.astype(grad_a, ct.bfloat16))
    ct.store(out_b_ptr, index=(n_idx, hw_idx, c_block),
             tile=ct.astype(grad_b, ct.bfloat16))


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="c8e869b1")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1,
        _shape_param_0, _shape_param_1,
    ) = inputs
    device = arg0_1.device

    # Channel-shuffle: build view_1 [N, 464, H, W] contiguous.
    # slice_1 = arg0[:, :232] (view/slice, no copy)
    slice_1 = arg0_1[:, :232]
    cat = torch.cat([slice_1, arg1_1], dim=1)  # [N, 464, H, W]
    view = cat.view(N, 232, 2, H, W)
    permuted = view.permute(0, 2, 1, 3, 4).contiguous()  # [N, 2, 232, H, W]
    view_1 = permuted.view(N, 464, H, W)
    slice_2 = view_1[:, 0:232]  # even
    slice_3 = view_1[:, 232:464]  # odd

    # Reshape channels-last inputs to [N, HW, C] contiguous.
    # arg2_1, arg8_1 have strides (11368, 1, 1624, 232) — channels-last.
    x_a_nhwc = arg2_1.permute(0, 2, 3, 1).reshape(N, HW, C)
    x_b_nhwc = arg8_1.permute(0, 2, 3, 1).reshape(N, HW, C)
    slice3_nhwc = slice_3.permute(0, 2, 3, 1).reshape(N, HW, C)
    slice2_nhwc = slice_2.permute(0, 2, 3, 1).reshape(N, HW, C)

    # Per-channel stats squeezed (arg3, arg4, arg9, arg10 have shape [1, C, 1, 1])
    mean_a = arg3_1.reshape(C).contiguous()
    invstd_a = arg4_1.reshape(C).contiguous()
    mean_b = arg9_1.reshape(C).contiguous()
    invstd_b = arg10_1.reshape(C).contiguous()
    weight_a = arg5_1.contiguous()
    weight_b = arg11_1.contiguous()
    bias_a = arg6_1.contiguous()
    bias_b = arg12_1.contiguous()
    zero_1 = arg7_1.contiguous().view(1)

    # Partial kernel launch config.
    BLOCK_N = 1
    BLOCK_HW = 64  # padded from 49
    BLOCK_C = 8
    num_hw_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW  # 1
    num_n_tiles = N // BLOCK_N  # 128
    NUM_TILES = num_n_tiles * num_hw_tiles  # 128
    num_c_tiles = (C + BLOCK_C - 1) // BLOCK_C  # 29

    partials = torch.zeros((4, C, NUM_TILES), device=device, dtype=torch.float32)
    partials_flat = partials.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUM_TILES, num_c_tiles, 1),
        _partial_kernel,
        (slice3_nhwc, slice2_nhwc, x_a_nhwc,
         mean_a, invstd_a, weight_a, bias_a, zero_1,
         x_b_nhwc, mean_b, invstd_b, weight_b, bias_b,
         partials_flat, NUM_TILES, BLOCK_N, BLOCK_HW, BLOCK_C),
    )

    sum_a = torch.empty(C, device=device, dtype=torch.float32)
    vec_a = torch.empty(C, device=device, dtype=torch.float32)
    sum_b = torch.empty(C, device=device, dtype=torch.float32)
    vec_b = torch.empty(C, device=device, dtype=torch.float32)
    stats = torch.empty(6 * C, device=device, dtype=torch.float32)

    FINAL_BLOCK_T = _next_power_of_2(NUM_TILES)
    FINAL_BLOCK_C = 8
    num_fc_tiles = (C + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C
    ct.launch(
        stream,
        (num_fc_tiles, 1, 1),
        _finalize_kernel,
        (partials_flat, invstd_a, weight_a, invstd_b, weight_b,
         sum_a, vec_a, sum_b, vec_b, stats,
         NUM_TILES, FINAL_BLOCK_T, FINAL_BLOCK_C),
    )

    # Prepare bf16 outputs channels-last.
    out_a = torch.empty_strided((N, C, H, W), (C * HW, 1, W * C, C),
                                device=device, dtype=torch.bfloat16)
    out_b = torch.empty_strided((N, C, H, W), (C * HW, 1, W * C, C),
                                device=device, dtype=torch.bfloat16)
    # View as [N, HW, C] contiguous
    out_a_nhwc = out_a.permute(0, 2, 3, 1).reshape(N, HW, C)
    out_b_nhwc = out_b.permute(0, 2, 3, 1).reshape(N, HW, C)

    EPI_BLOCK_N = 1
    EPI_BLOCK_HW = 64
    EPI_BLOCK_C = 8
    epi_hw_tiles = (HW + EPI_BLOCK_HW - 1) // EPI_BLOCK_HW
    epi_c_tiles = (C + EPI_BLOCK_C - 1) // EPI_BLOCK_C
    ct.launch(
        stream,
        (N, epi_hw_tiles, epi_c_tiles),
        _epilogue_kernel,
        (slice3_nhwc, slice2_nhwc, x_a_nhwc,
         mean_a, invstd_a, weight_a, bias_a, zero_1,
         x_b_nhwc, mean_b, invstd_b, weight_b, bias_b,
         stats, out_a_nhwc, out_b_nhwc,
         EPI_BLOCK_N, EPI_BLOCK_HW, EPI_BLOCK_C),
    )

    return sum_a, vec_a, out_a, sum_b, vec_b, out_b
