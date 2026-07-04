"""cuTile port of sum_sum_1f47655da3e5: Visformer avg-pool + BN-backward.

Matches Triton's three-kernel plan:
1. `_pool_bn_reduce_hw_kernel`: grid (cdiv(C, BLOCK_C), N). Computes
   partial_sum and partial_dot per (n, channel_block).
2. `_pool_bn_finalize_kernel`: grid (cdiv(C, BLOCK_C),). Reduces the
   partials across N and computes stats (mean_term / var / output_scale) plus
   sum_out and scale_grad output vectors.
3. `_pool_bn_epilogue_kernel`: grid (cdiv(C, BLOCK_C), N). Applies BN
   backward and writes out_f32 and out_bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 768
H = 7
W = 7
HW = H * W  # 49
DIVISOR = 49.0
REDUCTION_SCALE = 0.00015943877551020407
BLOCK_N = 128


@ct.kernel
def _pool_bn_reduce_hw_kernel(
    pooled_ptr,        # bf16 [N, C]
    activation_ptr,    # f32 [N, C, H, W]  (channels-last physically)
    partial_sum_ptr,   # f32 [N, C]
    partial_dot_ptr,   # f32 [N, C]
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    n_idx = ct.bid(1)
    hw = ct.arange(BLOCK_HW, dtype=ct.int32)
    c = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    hw_bcast = ct.reshape(hw, (BLOCK_HW, 1))
    c_bcast = ct.reshape(c, (1, BLOCK_C))
    HW_full = ct.full((BLOCK_HW, 1), HW, dtype=ct.int32)
    C_full = ct.full((1, BLOCK_C), C, dtype=ct.int32)
    mask = (hw_bcast < HW_full) & (c_bcast < C_full)

    h = hw // W
    w = hw - h * W
    h_2d = ct.reshape(h, (BLOCK_HW, 1))
    w_2d = ct.reshape(w, (BLOCK_HW, 1))
    zero_i2d = ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)
    n_2d = ct.full((BLOCK_HW, BLOCK_C), n_idx, dtype=ct.int32)
    h_full = h_2d + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)
    w_full = w_2d + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)
    c_full = c_bcast + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)

    n_safe = ct.where(mask, n_2d, zero_i2d)
    c_safe = ct.where(mask, c_full, zero_i2d)
    h_safe = ct.where(mask, h_full, zero_i2d)
    w_safe = ct.where(mask, w_full, zero_i2d)

    activation = ct.gather(
        activation_ptr, (n_safe, c_safe, h_safe, w_safe), mask=mask,
    )
    activation_f = activation

    # Load pooled[n, c] as a 1D (BLOCK_C,) tile, broadcast.
    pooled_c = ct.full((BLOCK_C,), 0.0, dtype=ct.bfloat16) + ct.gather(
        pooled_ptr,
        (ct.full((BLOCK_C,), n_idx, dtype=ct.int32),
         ct.where(c < ct.full((BLOCK_C,), C, dtype=ct.int32), c,
                  ct.zeros((BLOCK_C,), dtype=ct.int32))),
        mask=c < ct.full((BLOCK_C,), C, dtype=ct.int32),
    )
    pooled_f = ct.astype(pooled_c, ct.float32)
    pool_grad_1d = pooled_f * (1.0 / DIVISOR)
    pool_grad = ct.reshape(pool_grad_1d, (1, BLOCK_C)) + ct.zeros(
        (BLOCK_HW, BLOCK_C), dtype=ct.float32
    )

    zero_f2d = ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.float32)
    partial_sum = ct.sum(ct.where(mask, pool_grad, zero_f2d), axis=0)
    partial_dot = ct.sum(
        ct.where(mask, pool_grad * activation_f, zero_f2d), axis=0
    )

    c_mask = c < ct.full((BLOCK_C,), C, dtype=ct.int32)
    n_1d = ct.full((BLOCK_C,), n_idx, dtype=ct.int32)
    c_safe_1d = ct.where(c_mask, c, ct.zeros((BLOCK_C,), dtype=ct.int32))
    ct.scatter(
        partial_sum_ptr, (n_1d, c_safe_1d), partial_sum, mask=c_mask,
    )
    ct.scatter(
        partial_dot_ptr, (n_1d, c_safe_1d), partial_dot, mask=c_mask,
    )


@ct.kernel
def _pool_bn_finalize_kernel(
    partial_sum_ptr,   # f32 [N, C]
    partial_dot_ptr,   # f32 [N, C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    scale_grad_ptr,    # f32 [C]
    stats_ptr,         # f32 [3, C]   (mean_term, variance_term, output_scale)
    BLOCK_N_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    n = ct.arange(BLOCK_N_, dtype=ct.int32)
    c = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    n_2d = ct.reshape(n, (BLOCK_N_, 1))
    c_2d = ct.reshape(c, (1, BLOCK_C))
    N_full = ct.full((BLOCK_N_, 1), N, dtype=ct.int32)
    C_full = ct.full((1, BLOCK_C), C, dtype=ct.int32)
    mask = (n_2d < N_full) & (c_2d < C_full)
    zero_i2d = ct.zeros((BLOCK_N_, BLOCK_C), dtype=ct.int32)
    zero_f2d = ct.zeros((BLOCK_N_, BLOCK_C), dtype=ct.float32)

    n_full = n_2d + zero_i2d
    c_full = c_2d + zero_i2d
    n_safe = ct.where(mask, n_full, zero_i2d)
    c_safe = ct.where(mask, c_full, zero_i2d)

    partial_sum = ct.gather(
        partial_sum_ptr, (n_safe, c_safe), mask=mask,
    )
    partial_dot = ct.gather(
        partial_dot_ptr, (n_safe, c_safe), mask=mask,
    )
    partial_sum = ct.where(mask, partial_sum, zero_f2d)
    partial_dot = ct.where(mask, partial_dot, zero_f2d)
    sum_value = ct.sum(partial_sum, axis=0)
    dot_value = ct.sum(partial_dot, axis=0)

    c_mask = c < ct.full((BLOCK_C,), C, dtype=ct.int32)
    c_safe_1d = ct.where(c_mask, c, ct.zeros((BLOCK_C,), dtype=ct.int32))
    invstd = ct.gather(invstd_ptr, c_safe_1d, mask=c_mask)
    weight = ct.gather(weight_ptr, c_safe_1d, mask=c_mask)

    mean_term = sum_value * REDUCTION_SCALE
    dot_scaled = dot_value * REDUCTION_SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight
    scale_grad = dot_value * invstd

    ct.scatter(sum_out_ptr, c_safe_1d, sum_value, mask=c_mask)
    ct.scatter(scale_grad_ptr, c_safe_1d, scale_grad, mask=c_mask)

    # stats[0, c] = mean_term
    z0 = ct.zeros((BLOCK_C,), dtype=ct.int32)
    z1 = ct.full((BLOCK_C,), 1, dtype=ct.int32)
    z2 = ct.full((BLOCK_C,), 2, dtype=ct.int32)
    ct.scatter(stats_ptr, (z0, c_safe_1d), mean_term, mask=c_mask)
    ct.scatter(stats_ptr, (z1, c_safe_1d), variance_term, mask=c_mask)
    ct.scatter(stats_ptr, (z2, c_safe_1d), output_scale, mask=c_mask)


@ct.kernel
def _pool_bn_epilogue_kernel(
    pooled_ptr,        # bf16 [N, C]
    activation_ptr,    # f32 [N, C, H, W]
    stats_ptr,         # f32 [3, C]
    out_f32_ptr,       # f32 [N, C, H, W]
    out_bf16_ptr,      # bf16 [N, C, H, W]
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    n_idx = ct.bid(1)
    c = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    hw = ct.arange(BLOCK_HW, dtype=ct.int32)
    c_2d = ct.reshape(c, (BLOCK_C, 1))
    hw_2d = ct.reshape(hw, (1, BLOCK_HW))
    C_full = ct.full((BLOCK_C, 1), C, dtype=ct.int32)
    HW_full = ct.full((1, BLOCK_HW), HW, dtype=ct.int32)
    mask = (c_2d < C_full) & (hw_2d < HW_full)

    h = hw // W
    w = hw - h * W
    h_2d = ct.reshape(h, (1, BLOCK_HW))
    w_2d = ct.reshape(w, (1, BLOCK_HW))
    zero_i2d = ct.zeros((BLOCK_C, BLOCK_HW), dtype=ct.int32)
    n_2d = ct.full((BLOCK_C, BLOCK_HW), n_idx, dtype=ct.int32)
    h_full = h_2d + zero_i2d
    w_full = w_2d + zero_i2d
    c_full = c_2d + zero_i2d
    n_safe = ct.where(mask, n_2d, zero_i2d)
    c_safe = ct.where(mask, c_full, zero_i2d)
    h_safe = ct.where(mask, h_full, zero_i2d)
    w_safe = ct.where(mask, w_full, zero_i2d)

    activation = ct.gather(
        activation_ptr, (n_safe, c_safe, h_safe, w_safe), mask=mask,
    )

    c_1d_mask = c < ct.full((BLOCK_C,), C, dtype=ct.int32)
    c_safe_1d = ct.where(c_1d_mask, c, ct.zeros((BLOCK_C,), dtype=ct.int32))
    pooled_c = ct.gather(
        pooled_ptr,
        (ct.full((BLOCK_C,), n_idx, dtype=ct.int32), c_safe_1d),
        mask=c_1d_mask,
    )
    pooled_f = ct.astype(pooled_c, ct.float32)
    pool_grad_1d = pooled_f * (1.0 / DIVISOR)
    pool_grad = ct.reshape(pool_grad_1d, (BLOCK_C, 1)) + ct.zeros(
        (BLOCK_C, BLOCK_HW), dtype=ct.float32
    )

    mean_term_1d = ct.gather(
        stats_ptr,
        (ct.zeros((BLOCK_C,), dtype=ct.int32), c_safe_1d),
        mask=c_1d_mask,
    )
    variance_term_1d = ct.gather(
        stats_ptr,
        (ct.full((BLOCK_C,), 1, dtype=ct.int32), c_safe_1d),
        mask=c_1d_mask,
    )
    output_scale_1d = ct.gather(
        stats_ptr,
        (ct.full((BLOCK_C,), 2, dtype=ct.int32), c_safe_1d),
        mask=c_1d_mask,
    )
    mean_term = ct.reshape(mean_term_1d, (BLOCK_C, 1)) + ct.zeros(
        (BLOCK_C, BLOCK_HW), dtype=ct.float32,
    )
    variance_term = ct.reshape(variance_term_1d, (BLOCK_C, 1)) + ct.zeros(
        (BLOCK_C, BLOCK_HW), dtype=ct.float32,
    )
    output_scale = ct.reshape(output_scale_1d, (BLOCK_C, 1)) + ct.zeros(
        (BLOCK_C, BLOCK_HW), dtype=ct.float32,
    )

    after_variance = pool_grad - activation * variance_term
    after_mean = after_variance - mean_term
    out_f32 = after_mean * output_scale

    ct.scatter(
        out_f32_ptr, (n_safe, c_safe, h_safe, w_safe), out_f32, mask=mask,
    )
    ct.scatter(
        out_bf16_ptr, (n_safe, c_safe, h_safe, w_safe),
        ct.astype(out_f32, ct.bfloat16), mask=mask,
    )


@oracle_impl(hardware="B200", point="09c2ade4", BLOCK_HW=64, BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int):
    pooled, activation, invstd, weight, *_shape_params = inputs
    device = activation.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out_f32 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.float32,
    )
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty_strided((N, C), (C, 1), device=device, dtype=torch.float32)
    partial_dot = torch.empty_strided((N, C), (C, 1), device=device, dtype=torch.float32)
    stats = torch.empty_strided((3, C), (C, 1), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    grid_reduce = (ct.cdiv(C, BLOCK_C), N, 1)
    ct.launch(
        stream, grid_reduce, _pool_bn_reduce_hw_kernel,
        (pooled, activation, partial_sum, partial_dot, BLOCK_HW, BLOCK_C),
    )
    ct.launch(
        stream, (ct.cdiv(C, BLOCK_C), 1, 1), _pool_bn_finalize_kernel,
        (partial_sum, partial_dot, invstd, weight, sum_out, scale_grad, stats,
         BLOCK_N, BLOCK_C),
    )
    ct.launch(
        stream, grid_reduce, _pool_bn_epilogue_kernel,
        (pooled, activation, stats, out_f32, out_bf16, BLOCK_HW, BLOCK_C),
    )
    return sum_out, out_f32, scale_grad, out_bf16
