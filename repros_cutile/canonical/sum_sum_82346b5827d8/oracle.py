"""cuTile port of sum_sum_82346b5827d8: MNASNet BN-backward producer+finalize.

N=32, C=1280, H=W=7 (HW=49, non-power-of-2). Uses a per-N per-C-block reduction
kernel that produces `sum_1` and `sum_2` (dim=[0,2,3]) into partial buffers,
then a finalize kernel that combines partials and writes final scalars and dense
epilogue. Channels-last activation layout with stride (C*HW, 1, W*C, C).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 32
C = 1280
H = 7
W = 7
HW = H * W  # 49
BLOCK_HW = 64          # pow2 >= 49
BLOCK_C = 64
REDUCTION_SCALE = 0.0006377551020408163
INV_DROPOUT_KEEP = 1.25
INV_HW = 1.0 / 49.0


@ct.kernel
def _partial_reduce_kernel(
    mask2d_ptr,        # b8 [N, C]
    pooled_ptr,        # bf16 [N, C]
    x_ptr,             # bf16 [N, C, H, W] channels-last (stride (C*HW, 1, W*C, C))
    mean_ptr,          # f32 [C] (raw storage 1x1280x1x1)
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    bias_ptr,          # f32 [C]
    partial_sum_ptr,   # f32 [N, C]
    partial_dot_ptr,   # f32 [N, C]
    HIDDEN_: ct.Constant[int],       # HW
    BLOCK_HW_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    HW_: ct.Constant[int],
    W_: ct.Constant[int],
):
    n_idx = ct.bid(0)
    c_block = ct.bid(1)

    # Column vector [BLOCK_C]
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C_,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C_,))

    # mask2d and pooled at (n_idx, c_block) => shape (1, BLOCK_C)
    mask_bool = ct.load(mask2d_ptr, index=(n_idx, c_block), shape=(1, BLOCK_C_))
    pooled_bf16 = ct.load(pooled_ptr, index=(n_idx, c_block), shape=(1, BLOCK_C_))

    # pool_grad = pooled_bf16 * (mask_bool/0.8) / 49, all bf16 rounded
    mask_f = ct.astype(mask_bool, ct.float32)
    keep = ct.astype(mask_f * INV_DROPOUT_KEEP, ct.bfloat16)
    scaled = ct.astype(ct.astype(pooled_bf16, ct.float32) * ct.astype(keep, ct.float32), ct.bfloat16)
    pool_grad_bf16 = ct.astype(ct.astype(scaled, ct.float32) * INV_HW, ct.bfloat16)
    pool_grad_f = ct.astype(pool_grad_bf16, ct.float32)   # (1, BLOCK_C)

    # x is bf16 [N, C, H, W] channels-last -- stride (C*HW, 1, W*C, C).
    # Since it's channels-last, a natural cuTile view is (N, H, W, C) with
    # contiguous stride, but the harness treats it as (N, C, H, W). We load
    # from the true underlying storage manually. The channels-last layout means
    # element (n, c, h, w) has offset n*C*HW + h*W*C + w*C + c.
    # We iterate BLOCK_HW positions.

    # Zero accumulators
    acc_sum = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    acc_dot = ct.zeros((BLOCK_C_,), dtype=ct.float32)

    # For channels-last tensor viewed as [N, HW, C] (H*W flattened)
    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    hw_mask = hw_idx < HW_
    # Loop over BLOCK_HW rows; use hw indices
    # We'll do the reduction over hw axis (axis=0 in (BLOCK_HW, BLOCK_C) layout)

    # Load x[n_idx, :, :, :]'s BLOCK_HW rows across BLOCK_C channels.
    # Channels-last means adjacent memory in c-dim, so tile shape (BLOCK_HW, BLOCK_C)
    # over (HW, C) matches nicely.
    # x viewed as (N, HW, C)
    x_tile = ct.load(
        x_ptr, index=(n_idx, 0, c_block), shape=(1, BLOCK_HW_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_2d = ct.reshape(x_tile, (BLOCK_HW_, BLOCK_C_))
    x_f = ct.astype(x_2d, ct.float32)

    # Broadcast mean, invstd, weight, bias to (1, BLOCK_C)
    mean_row = ct.reshape(mean, (1, BLOCK_C_))
    invstd_row = ct.reshape(invstd, (1, BLOCK_C_))
    weight_row = ct.reshape(weight, (1, BLOCK_C_))
    bias_row = ct.reshape(bias, (1, BLOCK_C_))

    centered = x_f - mean_row
    normalized = centered * invstd_row
    weighted = normalized * weight_row
    affine = weighted + bias_row
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    # relu -> le 0 mask
    zeros_2d = ct.full((BLOCK_HW_, BLOCK_C_), 0.0, dtype=ct.bfloat16)
    le_zero = affine_bf16 <= zeros_2d

    # selected = where(le_zero, 0.0, pool_grad broadcast)
    pool_grad_row = ct.reshape(pool_grad_f, (1, BLOCK_C_))
    pool_grad_2d = pool_grad_row + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    selected = ct.where(le_zero, ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32), pool_grad_2d)

    # Mask by hw < HW_ (build 2D mask via product with ones, using int and cast)
    hw_row_mask_i = ct.reshape(ct.astype(hw_mask, ct.int32), (BLOCK_HW_, 1))
    hw_mask_2d_i = hw_row_mask_i + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.int32)
    hw_mask_2d = hw_mask_2d_i != 0
    selected = ct.where(hw_mask_2d, selected, 0.0)
    centered_masked = ct.where(hw_mask_2d, centered, 0.0)

    acc_sum = ct.sum(selected, axis=0)
    acc_dot = ct.sum(selected * centered_masked, axis=0)

    ct.store(partial_sum_ptr, index=(n_idx, c_block), tile=ct.reshape(acc_sum, (1, BLOCK_C_)))
    ct.store(partial_dot_ptr, index=(n_idx, c_block), tile=ct.reshape(acc_dot, (1, BLOCK_C_)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [N, C]
    partial_dot_ptr,   # f32 [N, C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    scaled_dot_out_ptr, # f32 [C]
    stats_ptr,         # f32 [3, C] -- mean_term, correction_scale, output_scale
    BLOCK_N_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    N_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c_block = ct.bid(0)

    # load partial across N rows
    tile = ct.load(
        partial_sum_ptr, index=(0, c_block), shape=(BLOCK_N_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile2 = ct.load(
        partial_dot_ptr, index=(0, c_block), shape=(BLOCK_N_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # Mask N rows > N_
    n_idx = ct.arange(BLOCK_N_, dtype=ct.int32)
    n_mask = n_idx < N_
    n_mask_2d_i = ct.reshape(ct.astype(n_mask, ct.int32), (BLOCK_N_, 1)) + ct.zeros((BLOCK_N_, BLOCK_C_), dtype=ct.int32)
    n_mask_2d = n_mask_2d_i != 0
    tile = ct.where(n_mask_2d, tile, 0.0)
    tile2 = ct.where(n_mask_2d, tile2, 0.0)

    sum_value = ct.sum(tile, axis=0)  # (BLOCK_C,)
    dot_value = ct.sum(tile2, axis=0)  # (BLOCK_C,)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,))
    mean_term = sum_value * SCALE_
    dot_mean = dot_value * SCALE_
    invstd_sq = invstd * invstd
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(scaled_dot_out_ptr, index=(c_block,), tile=dot_value * invstd)
    ct.store(stats_ptr, index=(0, c_block), tile=ct.reshape(mean_term, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(1, c_block), tile=ct.reshape(correction_scale, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(2, c_block), tile=ct.reshape(output_scale, (1, BLOCK_C_)))


@ct.kernel
def _epilogue_kernel(
    mask2d_ptr,        # b8 [N, C]
    pooled_ptr,        # bf16 [N, C]
    x_ptr,             # bf16 [N, HW, C] view
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    stats_ptr,         # f32 [3, C]
    out_ptr,           # bf16 [N, HW, C]
    BLOCK_HW_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    HW_: ct.Constant[int],
):
    n_idx = ct.bid(0)
    c_block = ct.bid(1)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C_,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C_,))

    mask_bool = ct.load(mask2d_ptr, index=(n_idx, c_block), shape=(1, BLOCK_C_))
    pooled_bf16 = ct.load(pooled_ptr, index=(n_idx, c_block), shape=(1, BLOCK_C_))

    mask_f = ct.astype(mask_bool, ct.float32)
    keep = ct.astype(mask_f * INV_DROPOUT_KEEP, ct.bfloat16)
    scaled = ct.astype(ct.astype(pooled_bf16, ct.float32) * ct.astype(keep, ct.float32), ct.bfloat16)
    pool_grad_bf16 = ct.astype(ct.astype(scaled, ct.float32) * INV_HW, ct.bfloat16)
    pool_grad_f = ct.astype(pool_grad_bf16, ct.float32)   # (1, BLOCK_C)

    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    hw_mask = hw_idx < HW_

    x_tile = ct.load(
        x_ptr, index=(n_idx, 0, c_block), shape=(1, BLOCK_HW_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_2d = ct.reshape(x_tile, (BLOCK_HW_, BLOCK_C_))
    x_f = ct.astype(x_2d, ct.float32)

    mean_row = ct.reshape(mean, (1, BLOCK_C_))
    invstd_row = ct.reshape(invstd, (1, BLOCK_C_))
    weight_row = ct.reshape(weight, (1, BLOCK_C_))
    bias_row = ct.reshape(bias, (1, BLOCK_C_))

    centered = x_f - mean_row
    normalized = centered * invstd_row
    weighted = normalized * weight_row
    affine = weighted + bias_row
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    zeros_2d = ct.full((BLOCK_HW_, BLOCK_C_), 0.0, dtype=ct.bfloat16)
    le_zero = affine_bf16 <= zeros_2d

    pool_grad_row = ct.reshape(pool_grad_f, (1, BLOCK_C_))
    pool_grad_2d = pool_grad_row + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    selected = ct.where(le_zero, ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32), pool_grad_2d)

    hw_row_mask_i = ct.reshape(ct.astype(hw_mask, ct.int32), (BLOCK_HW_, 1))
    hw_mask_2d_i = hw_row_mask_i + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.int32)
    hw_mask_2d = hw_mask_2d_i != 0
    selected = ct.where(hw_mask_2d, selected, 0.0)

    mean_term = ct.load(stats_ptr, index=(0, c_block), shape=(1, BLOCK_C_))
    correction_scale = ct.load(stats_ptr, index=(1, c_block), shape=(1, BLOCK_C_))
    output_scale = ct.load(stats_ptr, index=(2, c_block), shape=(1, BLOCK_C_))
    mean_term_2d = mean_term + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    correction_scale_2d = correction_scale + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    output_scale_2d = output_scale + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)

    correction = centered * correction_scale_2d
    after_correction = selected - correction
    after_mean = after_correction - mean_term_2d
    dense = ct.astype(after_mean * output_scale_2d, ct.bfloat16)

    # Store the tile. We must be careful with masked stores. Since HW=49 <
    # BLOCK_HW=64 we might over-write. We use ct.scatter for masked writes,
    # or write to a padded output then narrow.
    ct.store(out_ptr, index=(n_idx, 0, c_block), tile=ct.reshape(dense, (1, BLOCK_HW_, BLOCK_C_)))


@oracle_impl(hardware="B200", point="36943a14")
def oracle_forward(inputs):
    mask2d, pooled, x, mean, invstd, weight, bias, _shape0 = inputs
    device = x.device

    # Views for cuTile (channels-last -> (N, HW, C) with contiguous stride)
    # Channels-last (32, 1280, 7, 7) has stride (62720, 1, 8960, 1280).
    # A [N, HW, C] channels-last flattened view: element (n, h, w, c) offset
    # = n*C*HW + h*W*C + w*C + c = n*C*HW + hw*C + c. Stride (C*HW, C, 1).
    # We can create a (N, HW, C) contiguous view via as_strided.
    x_nhwc = x.as_strided((N, HW, C), (C * HW, C, 1))

    # mean, invstd are shape (1, 1280, 1, 1). Flatten to (C,).
    mean_1d = mean.view(C)
    invstd_1d = invstd.view(C)

    partial_sum = torch.empty((N, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((N, C), device=device, dtype=torch.float32)
    stats = torch.empty((3, C), device=device, dtype=torch.float32)
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    # dense_out with channels-last stride == (N, HW, C) view
    dense_nhwc = dense_out.as_strided((N, HW, C), (C * HW, C, 1))
    # Padded dense buffer (N, BLOCK_HW, C) for masked stores
    dense_pad = torch.empty((N, BLOCK_HW, C), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, ct.cdiv(C, BLOCK_C), 1),
        _partial_reduce_kernel,
        (mask2d, pooled, x_nhwc, mean_1d, invstd_1d, weight, bias,
         partial_sum, partial_dot, HW, BLOCK_HW, BLOCK_C, HW, W),
    )
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight,
         sum_out, scaled_dot_out, stats,
         64, BLOCK_C, N, REDUCTION_SCALE),
    )
    ct.launch(
        stream,
        (N, ct.cdiv(C, BLOCK_C), 1),
        _epilogue_kernel,
        (mask2d, pooled, x_nhwc, mean_1d, invstd_1d, weight, bias,
         stats, dense_pad, BLOCK_HW, BLOCK_C, HW),
    )
    # Copy valid (:, :HW, :) region into dense_nhwc
    dense_nhwc.copy_(dense_pad.narrow(1, 0, HW))

    # Return: (full=0.0 scalar bf16, sum_1, mul_11 (= scaled_dot_out * squeeze_1 == sum_2 * invstd_flat), dense_out)
    full = torch.zeros((), device=device, dtype=torch.bfloat16)
    return full, sum_out, scaled_dot_out, dense_out
