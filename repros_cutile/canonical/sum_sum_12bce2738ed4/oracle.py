"""cuTile port of sum_sum_12bce2738ed4: MobileNetV3 pool-backward + BN.

Two cuTile kernels, matching Triton's structure:
  1. reduce kernel: reads the pooled gradient (N, C), expands to per-pixel,
     applies hardswish backward + BN forward mask, then accumulates the
     per-channel sum + dot sums (in-kernel reduction).
  2. epilogue kernel: reads the finalized channel sums and re-produces the
     per-element BN-backward gradient tile-by-tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
C = 960
H = 7
W = 7
HW = H * W  # 49
R = BATCH * HW  # 1568
INV_N = 0.0006377551020408163  # 1/1568


@ct.kernel
def _direct_reduce_kernel(
    pooled_ptr,        # bf16 (N, C)
    x_ptr,             # bf16 (R, C)  channels-last flat
    mean_ptr,          # f32 (C,)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    bias_ptr,          # f32 (C,)
    fill_ptr,          # f32 (1,)
    sum_out_ptr,       # f32 (C,)
    dot_tmp_ptr,       # f32 (C,)
    scaled_dot_out_ptr, # f32 (C,)
    R_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    # (BLOCK_K rows, BLOCK_C cols) tile in the (R, C) layout.
    x_bf = ct.load(
        x_ptr, index=(0, c_block), shape=(BLOCK_K, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))

    x_f = ct.astype(x_bf, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    fill_f = ct.reshape(fill_tile, (1, 1))

    # Load pooled (N, C) values, expand-broadcast into the (BLOCK_K, BLOCK_C) tile.
    # Rows k = 0..BLOCK_K-1 map to n = k // HW.
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    n = k_idx // HW_
    k_active = k_idx < R_
    n_safe = ct.where(k_active, n, ct.zeros((BLOCK_K,), dtype=ct.int32))
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    c_active = c_idx < C
    c_safe = ct.where(c_active, c_idx, ct.zeros((BLOCK_C,), dtype=ct.int32))
    # Gather pooled[n, c] into (BLOCK_K, BLOCK_C).
    n_grid = ct.expand_dims(n_safe, 1)  # (BLOCK_K, 1)
    c_grid = ct.expand_dims(c_safe, 0)  # (1, BLOCK_C)
    n_grid2 = n_grid + ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.int32)
    c_grid2 = c_grid + ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.int32)
    pooled_bf = ct.gather(pooled_ptr, (n_grid2, c_grid2))
    pooled_f = ct.astype(pooled_bf, ct.float32)

    # pool_grad = bf16(pooled / 49.0)
    pool_grad = ct.astype(ct.astype(pooled_f / 49.0, ct.bfloat16), ct.float32)

    # BN forward affine: bf16((x - mean) * invstd * weight + bias)
    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    weighted = normalized * weight_2d
    affine_f = weighted + bias_2d
    affine_bf = ct.astype(affine_f, ct.bfloat16)
    affine_f2 = ct.astype(affine_bf, ct.float32)

    # hardswish backward mask:
    #   middle = pool_grad * (affine/3 + 0.5)
    #   producer = affine < 3 ? middle : pool_grad
    #   producer = affine <= -3 ? fill : producer
    #   producer_bf = bf16(producer)
    affine_third = affine_f2 / 3.0
    gate = affine_third + 0.5
    middle = pool_grad * gate
    lt_3 = affine_f2 < 3.0
    le_neg3 = affine_f2 <= -3.0
    producer = ct.where(lt_3, middle, pool_grad)
    fill_bcast = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.float32) + fill_f
    producer = ct.where(le_neg3, fill_bcast, producer)
    producer = ct.astype(ct.astype(producer, ct.bfloat16), ct.float32)

    # Mask OOB rows / cols to zero for the reduction.
    active_2d = ct.reshape(k_active, (BLOCK_K, 1)) & ct.reshape(c_active, (1, BLOCK_C))
    zero_2d = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.float32)
    producer_active = ct.where(active_2d, producer, zero_2d)
    centered_active = ct.where(active_2d, centered, zero_2d)
    dot = producer_active * centered_active

    sum_value = ct.sum(producer_active, axis=0)  # (BLOCK_C,)
    dot_value = ct.sum(dot, axis=0)              # (BLOCK_C,)

    # Only write valid columns.
    c_valid_mask = c_idx < C
    zero_c = ct.zeros((BLOCK_C,), dtype=ct.float32)
    sum_final = ct.where(c_valid_mask, sum_value, zero_c)
    dot_final = ct.where(c_valid_mask, dot_value, zero_c)
    scaled = dot_final * invstd

    ct.scatter(sum_out_ptr, (c_idx,), sum_final, mask=c_valid_mask)
    ct.scatter(dot_tmp_ptr, (c_idx,), dot_final, mask=c_valid_mask)
    ct.scatter(scaled_dot_out_ptr, (c_idx,), scaled, mask=c_valid_mask)


@ct.kernel
def _epilogue_kernel(
    pooled_ptr,        # bf16 (N, C)
    x_ptr,             # bf16 (R, C)
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    sum_ptr,           # f32 (C,)
    dot_ptr,           # f32 (C,)
    out_ptr,           # bf16 (R, C)
    R_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    row_block = ct.bid(0)
    c_block = ct.bid(1)

    x_bf = ct.load(
        x_ptr, index=(row_block, c_block), shape=(BLOCK_K, BLOCK_C),
    )
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    sum_1d = ct.load(sum_ptr, index=(c_block,), shape=(BLOCK_C,))
    dot_1d = ct.load(dot_ptr, index=(c_block,), shape=(BLOCK_C,))

    x_f = ct.astype(x_bf, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    fill_f = ct.reshape(fill_tile, (1, 1))

    # Reproduce producer.
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32) + row_block * BLOCK_K
    n = k_idx // HW_
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    n_grid = ct.expand_dims(n, 1) + ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.int32)
    c_grid = ct.expand_dims(c_idx, 0) + ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.int32)
    pooled_bf = ct.gather(pooled_ptr, (n_grid, c_grid))
    pooled_f = ct.astype(pooled_bf, ct.float32)

    pool_grad = ct.astype(ct.astype(pooled_f / 49.0, ct.bfloat16), ct.float32)

    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    weighted = normalized * weight_2d
    affine_f = weighted + bias_2d
    affine_bf = ct.astype(affine_f, ct.bfloat16)
    affine_f2 = ct.astype(affine_bf, ct.float32)

    affine_third = affine_f2 / 3.0
    gate = affine_third + 0.5
    middle = pool_grad * gate
    lt_3 = affine_f2 < 3.0
    le_neg3 = affine_f2 <= -3.0
    producer = ct.where(lt_3, middle, pool_grad)
    fill_bcast = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.float32) + fill_f
    producer = ct.where(le_neg3, fill_bcast, producer)
    producer = ct.astype(ct.astype(producer, ct.bfloat16), ct.float32)

    # Epilogue.
    sum_2d = ct.reshape(sum_1d, (1, BLOCK_C))
    dot_2d = ct.reshape(dot_1d, (1, BLOCK_C))
    mean_term = sum_2d * SCALE_
    dot_scaled = dot_2d * SCALE_
    invstd_sq = invstd_2d * invstd_2d
    var_term = dot_scaled * invstd_sq
    input_scale = invstd_2d * weight_2d
    without_var = producer - centered * var_term
    without_mean = without_var - mean_term
    out_f = without_mean * input_scale
    ct.store(out_ptr, index=(row_block, c_block), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="a7c9263b")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, _s0, _s1 = inputs
    device = arg1_1.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)

    # arg1_1 is channels-last with stride [C*HW, 1, W*C, C], i.e. NHWC contiguous.
    # We view it as (R, C) with R = N*HW.
    arg1_2d = arg1_1.permute(0, 2, 3, 1).reshape(R, C)
    # arg0_1 is (N, C).
    # Output is bf16 channels-last with same layout as arg1_1.
    grad_input = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=device,
        dtype=torch.bfloat16,
    )
    grad_input_2d = grad_input.permute(0, 2, 3, 1).view(R, C)

    mean_1d = arg2_1.view(C)
    invstd_1d = arg3_1.view(C)
    fill_1d = arg6_1.view(1)

    stream = torch.cuda.current_stream()

    BLOCK_K_REDUCE = 2048  # next pow2 >= R=1568
    BLOCK_C_REDUCE = 32
    ct.launch(
        stream, (ct.cdiv(C, BLOCK_C_REDUCE), 1, 1), _direct_reduce_kernel,
        (arg0_1, arg1_2d, mean_1d, invstd_1d, arg4_1, arg5_1, fill_1d,
         sum_out, dot_tmp, scaled_dot,
         R, HW, BLOCK_K_REDUCE, BLOCK_C_REDUCE),
    )

    BLOCK_K_EPI = 32
    BLOCK_C_EPI = 32
    # Need BLOCK_K_EPI to divide R=1568. 1568 = 2^5 * 7^2 = 32 * 49.
    # Use BLOCK_K_EPI=32 (divides 1568 as 49 tiles) and BLOCK_C_EPI=32 (divides 960 as 30).
    ct.launch(
        stream, (R // BLOCK_K_EPI, C // BLOCK_C_EPI, 1), _epilogue_kernel,
        (arg0_1, arg1_2d, mean_1d, invstd_1d, arg4_1, arg5_1, fill_1d,
         sum_out, dot_tmp, grad_input_2d,
         R, HW, BLOCK_K_EPI, BLOCK_C_EPI, INV_N),
    )
    return sum_out, scaled_dot, grad_input
