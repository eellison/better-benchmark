"""cuTile port of sum_sum_bbf2f7bed268: MobileNetV3 average-pool backward +
hard-sigmoid derivative + BN-backward.

Shapes: N=512, C=960, H=W=7 (channels-last). Same producer/reduce/epilogue
split as sum_sum_82346b5827d8 but with a scalar 'fill' input and hard-sigmoid
derivative logic instead of ReLU threshold. Non-power-of-2 dims handled with
padded output buffers and masked stores via narrow-copy.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 960
H = 7
W = 7
HW = H * W  # 49
BLOCK_HW = 64
BLOCK_C = 64
BLOCK_N = 64  # We'll reduce over N in chunks of BLOCK_N
REDUCTION_SCALE = 3.985969387755102e-05
INV_HW = 1.0 / 49.0


@ct.kernel
def _partial_reduce_kernel(
    pooled_ptr,     # bf16 [N, C] (squeezed from [N,C,1,1])
    x_ptr,          # bf16 [N, HW, C] channels-last view
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    fill_ptr,       # f32 [] scalar
    partial_sum_ptr, # f32 [N, C]
    partial_dot_ptr, # f32 [N, C]
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    n_idx = ct.bid(0)
    c_block = ct.bid(1)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C_,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C_,))
    pooled_bf16 = ct.load(pooled_ptr, index=(n_idx, c_block), shape=(1, BLOCK_C_))
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(ct.reshape(fill_scalar, (1,)), ct.float32)

    # pool_grad = pooled / 49 (bf16 rounded)
    pool_grad_bf16 = ct.astype(ct.astype(pooled_bf16, ct.float32) * INV_HW, ct.bfloat16)
    pool_grad_f = ct.astype(pool_grad_bf16, ct.float32)  # (1, BLOCK_C)

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
    affine_f = ct.astype(affine_bf16, ct.float32)

    # Hard-sigmoid derivative:
    # affine_third = affine_f / 3.0
    # gate = affine_third + 0.5
    # middle = pool_grad_f * gate
    # producer = where(affine_f < 3.0, middle, pool_grad_f)
    # producer = where(affine_f <= -3.0, fill, producer)
    affine_third = affine_f * (1.0 / 3.0)
    gate = affine_third + 0.5

    pool_grad_row = ct.reshape(pool_grad_f, (1, BLOCK_C_))
    pool_grad_2d = pool_grad_row + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    middle = pool_grad_2d * gate

    fill_2d = ct.reshape(fill_f, (1, 1)) + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)

    producer = ct.where(affine_f < 3.0, middle, pool_grad_2d)
    producer = ct.where(affine_f <= -3.0, fill_2d, producer)
    producer_bf16 = ct.astype(producer, ct.bfloat16)
    producer_f = ct.astype(producer_bf16, ct.float32)

    # Mask by hw < HW_
    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    hw_mask = hw_idx < HW_
    hw_row_mask_i = ct.reshape(ct.astype(hw_mask, ct.int32), (BLOCK_HW_, 1))
    hw_mask_2d_i = hw_row_mask_i + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.int32)
    hw_mask_2d = hw_mask_2d_i != 0

    producer_masked = ct.where(hw_mask_2d, producer_f, 0.0)
    centered_masked = ct.where(hw_mask_2d, centered, 0.0)

    acc_sum = ct.sum(producer_masked, axis=0)
    acc_dot = ct.sum(producer_masked * centered_masked, axis=0)

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
    stats_ptr,         # f32 [3, C]
    BLOCK_N_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    N_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c_block = ct.bid(0)

    # Since N=512 > BLOCK_N=64, we need to reduce across BLOCK_N tiles
    acc_sum = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    acc_dot = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    n_tiles = N_ // BLOCK_N_
    for t in range(n_tiles):
        tile = ct.load(
            partial_sum_ptr, index=(t, c_block), shape=(BLOCK_N_, BLOCK_C_),
        )
        tile2 = ct.load(
            partial_dot_ptr, index=(t, c_block), shape=(BLOCK_N_, BLOCK_C_),
        )
        acc_sum = acc_sum + ct.sum(tile, axis=0)
        acc_dot = acc_dot + ct.sum(tile2, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,))
    mean_term = acc_sum * SCALE_
    dot_mean = acc_dot * SCALE_
    invstd_sq = invstd * invstd
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=acc_sum)
    ct.store(scaled_dot_out_ptr, index=(c_block,), tile=acc_dot * invstd)
    ct.store(stats_ptr, index=(0, c_block), tile=ct.reshape(mean_term, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(1, c_block), tile=ct.reshape(correction_scale, (1, BLOCK_C_)))
    ct.store(stats_ptr, index=(2, c_block), tile=ct.reshape(output_scale, (1, BLOCK_C_)))


@ct.kernel
def _epilogue_kernel(
    pooled_ptr,        # bf16 [N, C]
    x_ptr,             # bf16 [N, HW, C]
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    stats_ptr,         # f32 [3, C]
    out_ptr,           # bf16 [N, BLOCK_HW, C] padded
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
    pooled_bf16 = ct.load(pooled_ptr, index=(n_idx, c_block), shape=(1, BLOCK_C_))
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(ct.reshape(fill_scalar, (1,)), ct.float32)

    pool_grad_bf16 = ct.astype(ct.astype(pooled_bf16, ct.float32) * INV_HW, ct.bfloat16)
    pool_grad_f = ct.astype(pool_grad_bf16, ct.float32)

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
    affine_f = ct.astype(affine_bf16, ct.float32)

    affine_third = affine_f * (1.0 / 3.0)
    gate = affine_third + 0.5

    pool_grad_row = ct.reshape(pool_grad_f, (1, BLOCK_C_))
    pool_grad_2d = pool_grad_row + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    middle = pool_grad_2d * gate

    fill_2d = ct.reshape(fill_f, (1, 1)) + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)

    producer = ct.where(affine_f < 3.0, middle, pool_grad_2d)
    producer = ct.where(affine_f <= -3.0, fill_2d, producer)
    producer_bf16 = ct.astype(producer, ct.bfloat16)
    producer_f = ct.astype(producer_bf16, ct.float32)

    mean_term = ct.load(stats_ptr, index=(0, c_block), shape=(1, BLOCK_C_))
    correction_scale = ct.load(stats_ptr, index=(1, c_block), shape=(1, BLOCK_C_))
    output_scale = ct.load(stats_ptr, index=(2, c_block), shape=(1, BLOCK_C_))
    mean_term_2d = mean_term + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    correction_scale_2d = correction_scale + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)
    output_scale_2d = output_scale + ct.zeros((BLOCK_HW_, BLOCK_C_), dtype=ct.float32)

    correction = centered * correction_scale_2d
    after_correction = producer_f - correction
    after_mean = after_correction - mean_term_2d
    dense = ct.astype(after_mean * output_scale_2d, ct.bfloat16)

    ct.store(out_ptr, index=(n_idx, 0, c_block), tile=ct.reshape(dense, (1, BLOCK_HW_, BLOCK_C_)))


@oracle_impl(
    hardware="B200", point="d90f6100",
    GROUP_K=256, BLOCK_K=256, BLOCK_C=16, OUT_BLOCK_R=None, FINAL_BLOCK_C=None,
)
def oracle_forward(inputs, **_kwargs):
    (
        pooled_bcnn,     # bf16 [N, C, 1, 1] (arg0)
        x_bchw,          # bf16 [N, C, H, W] channels-last (arg1)
        mean_bcnn,       # f32 [1, C, 1, 1] (arg2)
        invstd_bcnn,     # f32 [1, C, 1, 1] (arg3)
        weight_1d,       # f32 [C] (arg4)
        bias_1d,         # f32 [C] (arg5)
        fill_scalar,     # f32 [] (arg6)
        *_shape_params,
    ) = inputs
    device = x_bchw.device

    # pooled: (N,C,1,1) bf16 -> view as (N, C)
    pooled_2d = pooled_bcnn.view(N, C)

    # x is channels-last (N,C,H,W) stride (C*HW, 1, W*C, C).
    # View as (N, HW, C) with stride (C*HW, C, 1).
    x_nhwc = x_bchw.as_strided((N, HW, C), (C * HW, C, 1))

    mean_1d = mean_bcnn.view(C)
    invstd_1d = invstd_bcnn.view(C)
    fill_1d = fill_scalar.view(1)

    partial_sum = torch.empty((N, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((N, C), device=device, dtype=torch.float32)
    stats = torch.empty((3, C), device=device, dtype=torch.float32)
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    dense_nhwc = dense_out.as_strided((N, HW, C), (C * HW, C, 1))
    dense_pad = torch.empty((N, BLOCK_HW, C), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, ct.cdiv(C, BLOCK_C), 1),
        _partial_reduce_kernel,
        (pooled_2d, x_nhwc, mean_1d, invstd_1d, weight_1d, bias_1d, fill_1d,
         partial_sum, partial_dot, HW, BLOCK_HW, BLOCK_C),
    )
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, scaled_dot_out, stats, BLOCK_N, BLOCK_C, N, REDUCTION_SCALE),
    )
    ct.launch(
        stream,
        (N, ct.cdiv(C, BLOCK_C), 1),
        _epilogue_kernel,
        (pooled_2d, x_nhwc, mean_1d, invstd_1d, weight_1d, bias_1d, fill_1d,
         stats, dense_pad, BLOCK_HW, BLOCK_C, HW),
    )
    dense_nhwc.copy_(dense_pad.narrow(1, 0, HW))

    return sum_out, scaled_dot_out, dense_out
