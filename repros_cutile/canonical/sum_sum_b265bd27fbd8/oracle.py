"""cuTile port of sum_sum_b265bd27fbd8: GhostNet BN-backward channels-last split-K.

Fair port: three kernels mirroring Triton's `_partial_reduce_kernel`,
`_finalize_kernel`, `_epilogue_kernel`. All reductions live inside `@ct.kernel`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
FULL_C = 184
C = 92
H = 14
W = 14
HW = H * W
R = N * HW
OUT_NUMEL = N * C * HW
INV_R = 9.964923469387754e-06
SLICE_START = 92


def _ceil_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_reduce_kernel(
    slice_base_ptr,   # bf16 flat, channels-last NHWC
    bn_input_ptr,     # bf16 flat, channels-last NHWC (or strided; treat via gather)
    mean_ptr,         # f32 [C]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    bias_ptr,         # f32 [C]
    fill_ptr,         # bf16 [1]
    partial_sum_ptr,  # f32 (num_tiles * C_PAD,)
    partial_dot_ptr,  # f32 (num_tiles * C_PAD,)
    R_: ct.Constant[int],
    FULL_C_: ct.Constant[int],
    C_: ct.Constant[int],
    C_PAD: ct.Constant[int],
    HW_: ct.Constant[int],
    W_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile_r = ct.bid(0)
    tile_c = ct.bid(1)

    rows_1d = tile_r * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    cols_1d = tile_c * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = rows_1d < R_
    col_valid = cols_1d < C_

    n = rows_1d // HW_
    hw = rows_1d - n * HW_
    h = hw // W_
    w = hw - h * W_

    n_2d = ct.reshape(n, (BLOCK_R, 1))
    h_2d = ct.reshape(h, (BLOCK_R, 1))
    w_2d = ct.reshape(w, (BLOCK_R, 1))
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_C))
    zero_i = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)

    source_off = ((n_2d + zero_i) * (FULL_C_ * HW_)
                  + (h_2d + zero_i) * (W_ * FULL_C_)
                  + (w_2d + zero_i) * FULL_C_
                  + SLICE_START_
                  + (cols_2d + zero_i))
    compact_off = ((n_2d + zero_i) * (C_ * HW_)
                   + (h_2d + zero_i) * (W_ * C_)
                   + (w_2d + zero_i) * C_
                   + (cols_2d + zero_i))

    source_bf = ct.gather(slice_base_ptr, source_off)
    x_bf = ct.gather(bn_input_ptr, compact_off)
    x = ct.astype(x_bf, ct.float32)

    mean = ct.gather(mean_ptr, cols_2d + zero_i)
    invstd = ct.gather(invstd_ptr, cols_2d + zero_i)
    weight = ct.gather(weight_ptr, cols_2d + zero_i)
    bias = ct.gather(bias_ptr, cols_2d + zero_i)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)
    ones_bc = ct.full((BLOCK_R, BLOCK_C), 1.0, dtype=ct.float32)
    fill_bc_bf = ct.astype(ones_bc * fill_f, ct.bfloat16)

    centered = x - mean
    affine = centered * invstd * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.bfloat16)
    selected_bf = ct.where(affine_bf <= zero_bf, fill_bc_bf, source_bf)
    selected = ct.astype(selected_bf, ct.float32)

    row_valid_2d = ct.reshape(row_valid, (BLOCK_R, 1))
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_C))
    valid = row_valid_2d & col_valid_2d
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    selected_m = ct.where(valid, selected, zero_f)
    centered_m = ct.where(valid, centered, zero_f)

    p_sum = ct.sum(selected_m, axis=0)                    # (BLOCK_C,)
    p_dot = ct.sum(selected_m * centered_m, axis=0)       # (BLOCK_C,)

    p_off = tile_r * C_PAD + cols_1d
    ct.scatter(partial_sum_ptr, p_off, p_sum, mask=col_valid)
    ct.scatter(partial_dot_ptr, p_off, p_dot, mask=col_valid)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (BLOCK_TILES_PAD, C_PAD) 2D backing
    partial_dot_ptr,   # f32 (BLOCK_TILES_PAD, C_PAD) 2D backing
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    C_: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_R_: ct.Constant[float],
):
    c_block = ct.bid(0)
    # 2D load of full padded (BLOCK_TILES, BLOCK_C) tile — OOB zero-padded.
    p_sum = ct.load(
        partial_sum_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    p_dot = ct.load(
        partial_dot_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )

    tiles = ct.arange(BLOCK_TILES, dtype=ct.int32)
    cols_local = ct.arange(BLOCK_C, dtype=ct.int32)
    cols = c_block * BLOCK_C + cols_local
    col_valid = cols < C_
    tile_valid = tiles < NUM_TILES
    tile_valid_2d = ct.reshape(tile_valid, (BLOCK_TILES, 1))
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_C))
    valid = tile_valid_2d & col_valid_2d
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)
    p_sum_m = ct.where(valid, p_sum, zero_f)
    p_dot_m = ct.where(valid, p_dot, zero_f)

    s1 = ct.sum(p_sum_m, axis=0)
    s2 = ct.sum(p_dot_m, axis=0)

    invstd = ct.gather(invstd_ptr, cols)
    weight = ct.gather(weight_ptr, cols)
    dot_scaled = s2 * INV_R_
    invstd_sq = invstd * invstd

    ct.scatter(sum_out_ptr, cols, s1, mask=col_valid)
    ct.scatter(scale_grad_ptr, cols, s2 * invstd, mask=col_valid)
    ct.scatter(mean_term_ptr, cols, s1 * INV_R_, mask=col_valid)
    ct.scatter(dot_coeff_ptr, cols, dot_scaled * invstd_sq, mask=col_valid)
    ct.scatter(out_scale_ptr, cols, invstd * weight, mask=col_valid)


@ct.kernel
def _epilogue_kernel(
    slice_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    out_ptr,
    OUT_NUMEL_: ct.Constant[int],
    FULL_C_: ct.Constant[int],
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    W_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    BN_S0: ct.Constant[int],
    BN_S1: ct.Constant[int],
    BN_S2: ct.Constant[int],
    BN_S3: ct.Constant[int],
    OUT_S0: ct.Constant[int],
    OUT_S1: ct.Constant[int],
    OUT_S2: ct.Constant[int],
    OUT_S3: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    linear = ct.arange(BLOCK, dtype=ct.int64) + pid * BLOCK
    OUT_NUMEL_i = ct.full((BLOCK,), OUT_NUMEL_, dtype=ct.int64)
    valid = linear < OUT_NUMEL_i
    zero_i = ct.zeros((BLOCK,), dtype=ct.int64)
    safe_lin = ct.where(valid, linear, zero_i)

    C_i = ct.full((BLOCK,), C_, dtype=ct.int64)
    HW_i = ct.full((BLOCK,), HW_, dtype=ct.int64)
    W_i = ct.full((BLOCK,), W_, dtype=ct.int64)
    FULL_C_i = ct.full((BLOCK,), FULL_C_, dtype=ct.int64)
    SLICE_i = ct.full((BLOCK,), SLICE_START_, dtype=ct.int64)

    c = safe_lin - (safe_lin // C_i) * C_i
    row = safe_lin // C_i
    n = row // HW_i
    hw = row - n * HW_i
    h = hw // W_i
    w = hw - h * W_i

    source_offsets = n * (FULL_C_i * HW_i) + h * (W_i * FULL_C_i) + w * FULL_C_i + SLICE_i + c
    bn_offsets = n * BN_S0 + c * BN_S1 + h * BN_S2 + w * BN_S3
    out_offsets = n * OUT_S0 + c * OUT_S1 + h * OUT_S2 + w * OUT_S3

    source = ct.gather(slice_ptr, source_offsets)
    x = ct.astype(ct.gather(bn_input_ptr, bn_offsets), ct.float32)
    mean = ct.gather(mean_ptr, c)
    invstd = ct.gather(invstd_ptr, c)
    weight = ct.gather(weight_ptr, c)
    bias = ct.gather(bias_ptr, c)
    fill_val = ct.load(fill_ptr, index=(0,), shape=(1,))

    centered = x - mean
    affine = centered * invstd * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    selected_bf = ct.where(affine_bf <= zero_bf, fill_val, source)
    selected = ct.astype(selected_bf, ct.float32)

    mean_term = ct.gather(mean_term_ptr, c)
    dot_coeff = ct.gather(dot_coeff_ptr, c)
    out_scale = ct.gather(out_scale_ptr, c)
    adjusted = selected - centered * dot_coeff - mean_term
    out_val = adjusted * out_scale
    out_bf = ct.astype(out_val, ct.bfloat16)
    ct.scatter(out_ptr, out_offsets, out_bf, mask=valid)


@oracle_impl(hardware="B200", point="f636f756", BLOCK_R=512, BLOCK_C=4, FINAL_BLOCK_C=4)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    slice_base, bn_input, mean, invstd, weight, bias, fill = inputs
    device = bn_input.device

    mean_f = mean.view(C).to(torch.float32)
    invstd_f = invstd.view(C).to(torch.float32)
    weight_f = weight.view(C).to(torch.float32)
    bias_f = bias.view(C).to(torch.float32)

    num_tiles = (R + BLOCK_R - 1) // BLOCK_R
    block_tiles = _ceil_pow2(num_tiles)
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    num_final_blocks = (C + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C

    def _flat(t):
        size = t.untyped_storage().size() // 2
        return torch.as_strided(t, (size,), (1,), 0)

    slice_flat = _flat(slice_base)
    bn_flat = _flat(bn_input)

    # Allocate padded partials so both dims divide BLOCK sizes (avoids
    # cuTile misaligned-address errors on 2D loads).
    c_pad = num_final_blocks * FINAL_BLOCK_C  # divisible by FINAL_BLOCK_C
    partial_sum_2d = torch.zeros((block_tiles, c_pad), device=device, dtype=torch.float32)
    partial_dot_2d = torch.zeros((block_tiles, c_pad), device=device, dtype=torch.float32)
    # Flat views for the partial_reduce_kernel which uses 1D scatter offsets.
    partial_sum = partial_sum_2d.view(-1)
    partial_dot = partial_dot_2d.view(-1)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    dot_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_tiles, num_c_blocks, 1),
        _partial_reduce_kernel,
        (slice_flat, bn_flat, mean_f, invstd_f, weight_f, bias_f, fill.view(1),
         partial_sum, partial_dot,
         R, FULL_C, C, c_pad, HW, W, SLICE_START, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_final_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum_2d, partial_dot_2d, invstd_f, weight_f,
         sum_out, scale_grad, mean_term, dot_coeff, out_scale,
         C, num_tiles, block_tiles, FINAL_BLOCK_C, INV_R),
    )

    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    out_flat = _flat(dense_out)

    BLOCK = 512
    ct.launch(
        stream,
        (ct.cdiv(OUT_NUMEL, BLOCK), 1, 1),
        _epilogue_kernel,
        (slice_flat, bn_flat,
         mean_f, invstd_f, weight_f, bias_f, fill.view(1),
         mean_term, dot_coeff, out_scale, out_flat,
         OUT_NUMEL, FULL_C, C, HW, W, SLICE_START,
         int(bn_input.stride(0)), int(bn_input.stride(1)),
         int(bn_input.stride(2)), int(bn_input.stride(3)),
         int(dense_out.stride(0)), int(dense_out.stride(1)),
         int(dense_out.stride(2)), int(dense_out.stride(3)),
         BLOCK),
    )

    return sum_out, scale_grad, dense_out
