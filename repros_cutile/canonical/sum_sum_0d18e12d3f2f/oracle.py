"""cuTile port of sum_sum_0d18e12d3f2f: GhostNet bf16 masked BN-backward.

Mirrors the Triton oracle's 3-kernel structure:
  1. _partial_reduce_kernel: tile (BLOCK_R rows, BLOCK_C cols) → partials.
  2. _finalize_kernel: per-channel reduce over row tiles → final sums.
  3. _epilogue_kernel: per-pixel BN-backward gradient.

All in-kernel: masked producer (affine → relu <= 0 → where fill/source), and
sibling reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 72
C = 36
SLICE_START = 36
H = 56
W = 56
HW = H * W
R = N * HW  # 1605632
NUMEL = N * C * HW
INV_R = 6.228077168367346e-07


def _masked_where_and_centered(
    x_bf, source_bf, mean, invstd, weight, bias, fill, br, bc,
):
    """Compute the masked selected value (f32) and centered (f32).

    Broadcasts scalar/per-channel params over the (br, bc) tile.
    """
    x_f = ct.astype(x_bf, ct.float32)
    centered = x_f - mean
    normalized = centered * invstd
    weighted = normalized * weight
    affine_f = weighted + bias
    affine_bf = ct.astype(affine_f, ct.bfloat16)
    zero_bf = ct.zeros((br, bc), dtype=ct.bfloat16)
    le = affine_bf <= zero_bf
    selected = ct.where(le, fill, source_bf)
    return ct.astype(selected, ct.float32), centered


@ct.kernel
def _partial_reduce_kernel(
    src_ptr,           # bf16 (R, C_IN)  NHWC-flat (channels-last flat)
    mask_input_ptr,    # bf16 (R, C)     NHWC-flat
    mean_ptr,          # f32 (C,)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    bias_ptr,          # f32 (C,)
    fill_ptr,          # bf16 (1,)
    partial_sum_ptr,   # f32 (NUM_TILES, C)
    partial_dot_ptr,   # f32 (NUM_TILES, C)
    R_: ct.Constant[int],
    C_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)
    c_block = ct.bid(1)

    x_bf = ct.load(
        mask_input_ptr, index=(tile, c_block), shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # source is src[:, SLICE_START:SLICE_START+C, :, :] in NHWC → columns
    # [SLICE_START, SLICE_START+C) of src viewed as (R, C_IN).
    # We index in c-space by adding SLICE_START offset.
    # cuTile can't easily do column offset in tile-space; use gather.
    row_idx = ct.arange(BLOCK_R, dtype=ct.int32) + tile * BLOCK_R
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    row_active = row_idx < R_
    col_active = col_idx < C_
    active = ct.reshape(row_active, (BLOCK_R, 1)) & ct.reshape(col_active, (1, BLOCK_C))
    row_safe = ct.where(row_active, row_idx, ct.zeros((BLOCK_R,), dtype=ct.int32))
    col_src = ct.where(col_active, col_idx + SLICE_START_, ct.zeros((BLOCK_C,), dtype=ct.int32))
    row_g = ct.expand_dims(row_safe, 1) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    col_g = ct.expand_dims(col_src, 0) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    source_bf = ct.gather(src_ptr, (row_g, col_g))

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_2d = ct.reshape(fill_tile, (1, 1))

    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))

    where_f, centered = _masked_where_and_centered(
        x_bf, source_bf, mean_2d, invstd_2d, weight_2d, bias_2d, fill_2d, BLOCK_R, BLOCK_C,
    )
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    where_active = ct.where(active, where_f, zero_f)
    centered_active = ct.where(active, centered, zero_f)
    dot = where_active * centered_active

    sum_r = ct.sum(where_active, axis=0)
    dot_r = ct.sum(dot, axis=0)

    # Write to partial buffers (NUM_TILES, C) with per-column masking via scatter.
    zero_c = ct.zeros((BLOCK_C,), dtype=ct.float32)
    sum_out = ct.where(col_active, sum_r, zero_c)
    dot_out = ct.where(col_active, dot_r, zero_c)
    tile_row = ct.full((BLOCK_C,), tile, dtype=ct.int32)
    ct.scatter(partial_sum_ptr, (tile_row, col_idx), sum_out, mask=col_active)
    ct.scatter(partial_dot_ptr, (tile_row, col_idx), dot_out, mask=col_active)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (NUM_TILES, C)
    partial_dot_ptr,   # f32 (NUM_TILES, C)
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    NUM_TILES: ct.Constant[int],
    NUM_TILES_PAD: ct.Constant[int],
    C_: ct.Constant[int],
    INV_R_: ct.Constant[float],
):
    c = ct.bid(0)
    sum_partials = ct.load(
        partial_sum_ptr, index=(0, c), shape=(NUM_TILES_PAD, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    dot_partials = ct.load(
        partial_dot_ptr, index=(0, c), shape=(NUM_TILES_PAD, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_idx = ct.arange(NUM_TILES_PAD, dtype=ct.int32)
    active = ct.reshape(tile_idx < NUM_TILES, (NUM_TILES_PAD, 1))
    zero_2d = ct.zeros((NUM_TILES_PAD, 1), dtype=ct.float32)
    sum_masked = ct.where(active, sum_partials, zero_2d)
    dot_masked = ct.where(active, dot_partials, zero_2d)
    sum_value = ct.sum(sum_masked)  # 0-d
    dot_value = ct.sum(dot_masked)  # 0-d

    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    sum_tile = ct.full((1,), sum_value, dtype=ct.float32)
    dot_tile = ct.full((1,), dot_value, dtype=ct.float32)

    ct.store(sum_out_ptr, index=(c,), tile=sum_tile)
    ct.store(vec_out_ptr, index=(c,), tile=dot_tile * invstd)
    ct.store(mean_term_ptr, index=(c,), tile=sum_tile * INV_R_)
    dot_scaled = dot_tile * INV_R_
    invstd_sq = invstd * invstd
    ct.store(dot_coeff_ptr, index=(c,), tile=dot_scaled * invstd_sq)
    ct.store(out_scale_ptr, index=(c,), tile=invstd * weight)


@ct.kernel
def _epilogue_kernel(
    src_ptr,           # bf16 (R, C_IN)
    mask_input_ptr,    # bf16 (R, C)
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    mean_term_ptr,     # f32 (C,)
    dot_coeff_ptr,     # f32 (C,)
    out_scale_ptr,     # f32 (C,)
    out_ptr,           # bf16 (R, C)
    R_: ct.Constant[int],
    C_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    c_block = ct.bid(1)

    x_bf = ct.load(mask_input_ptr, index=(row_block, c_block), shape=(BLOCK_R, BLOCK_C))
    row_idx = ct.arange(BLOCK_R, dtype=ct.int32) + row_block * BLOCK_R
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    col_src = col_idx + SLICE_START_
    row_g = ct.expand_dims(row_idx, 1) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    col_g = ct.expand_dims(col_src, 0) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    source_bf = ct.gather(src_ptr, (row_g, col_g))

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_2d = ct.reshape(fill_tile, (1, 1))
    mean_term = ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C,))
    dot_coeff = ct.load(dot_coeff_ptr, index=(c_block,), shape=(BLOCK_C,))
    out_scale = ct.load(out_scale_ptr, index=(c_block,), shape=(BLOCK_C,))

    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    dot_coeff_2d = ct.reshape(dot_coeff, (1, BLOCK_C))
    out_scale_2d = ct.reshape(out_scale, (1, BLOCK_C))

    where_f, centered = _masked_where_and_centered(
        x_bf, source_bf, mean_2d, invstd_2d, weight_2d, bias_2d, fill_2d, BLOCK_R, BLOCK_C,
    )
    adjusted = where_f - centered * dot_coeff_2d - mean_term_2d
    out_f = adjusted * out_scale_2d
    ct.store(out_ptr, index=(row_block, c_block), tile=ct.astype(out_f, ct.bfloat16))


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(hardware="B200", point="ef0a0981", BLOCK_R=1024, BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    device = arg0_1.device

    # Both arg0_1 and arg1_1 are channels-last (NHWC) contiguous.
    # View as (R, C_IN) and (R, C).
    src_2d = arg0_1.permute(0, 2, 3, 1).reshape(R, C_IN)
    mask_2d = arg1_1.permute(0, 2, 3, 1).reshape(R, C)

    mean_1d = arg2_1.view(C)
    invstd_1d = arg3_1.view(C)
    fill_1d = arg6_1.view(1)

    # Choose BLOCK_R/BLOCK_C such that BLOCK_R divides R and BLOCK_C divides C so
    # the partial-reduce load has no OOB. R=1605632 = 512 * 3136; use BLOCK_R=512.
    # C=36 (not power of 2). Use BLOCK_C=4 (pow2 divisor of 36 → 9 col tiles).
    br = 512
    bc = 4
    assert R % br == 0
    num_tiles = R // br
    num_tiles_pad = _next_p2(num_tiles)

    partial_sum = torch.zeros((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.zeros((num_tiles, C), device=device, dtype=torch.float32)
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    vec_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    dot_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, C * W, C),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_out_2d = dense_out.permute(0, 2, 3, 1).view(R, C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_tiles, ct.cdiv(C, bc), 1), _partial_reduce_kernel,
        (src_2d, mask_2d, mean_1d, invstd_1d, arg4_1, arg5_1, fill_1d,
         partial_sum, partial_dot,
         R, C, SLICE_START, br, bc),
    )
    ct.launch(
        stream, (C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, arg4_1,
         sum_out, vec_out, mean_term, dot_coeff, out_scale,
         num_tiles, num_tiles_pad, C, INV_R),
    )
    ct.launch(
        stream, (num_tiles, ct.cdiv(C, bc), 1), _epilogue_kernel,
        (src_2d, mask_2d, mean_1d, invstd_1d, arg4_1, arg5_1, fill_1d,
         mean_term, dot_coeff, out_scale, dense_out_2d,
         R, C, SLICE_START, br, bc),
    )
    return sum_out, vec_out, dense_out
