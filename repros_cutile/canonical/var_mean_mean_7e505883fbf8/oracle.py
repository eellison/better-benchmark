"""cuTile port of var_mean_mean_7e505883fbf8: Visformer training-BN plus
spatial mean. Two kernels (skipping Welford by doing a single-pass mean + var
reduction directly per channel):
  1) BN stats: reduce K = N*H*W to per-channel mean/var; mutate running stats.
  2) Center + affine + spatial mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 768
H = 7
W = 7
HW = H * W       # 49
K = N * HW       # 6272

EPS = 1.0e-5
MOMENTUM = 0.1
OLD_WEIGHT = 0.9
RUNNING_VAR_CORRECTION = 1.0001594642002871


@ct.kernel
def _bn_stats_kernel(
    x0_ptr,    # f32 flat NHWC [N*C*H*W]
    x1_ptr,    # bf16 flat NHWC [N*C*H*W]
    running_mean_ptr,   # f32 [C]
    running_var_ptr,    # f32 [C]
    mean_out_ptr,       # f32 [C]
    invstd_out_ptr,     # f32 [C]
    BLOCK: ct.Constant[int],
):
    channel = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32)
    active = offsets < K
    n_idx = offsets // HW
    hw_idx = offsets - n_idx * HW
    # NHWC flat: n * (H * W * C) + hw * C + c   (since H*W = HW)
    x_offsets = n_idx * (HW * C) + hw_idx * C + channel

    x0 = ct.gather(x0_ptr, (x_offsets,), mask=active, padding_value=ct.float32(0.0))
    x1 = ct.astype(ct.gather(x1_ptr, (x_offsets,), mask=active, padding_value=ct.bfloat16(0.0)), ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    values = ct.where(active, x0 + x1, zero_f)
    mean = ct.sum(values) / float(K)
    centered = ct.where(active, values - mean, zero_f)
    var_val = ct.sum(centered * centered) / float(K)
    invstd = ct.rsqrt(var_val + EPS)

    ct.store(mean_out_ptr, index=(channel,),
              tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + mean, (1,)))
    ct.store(invstd_out_ptr, index=(channel,),
              tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + invstd, (1,)))

    # Running stats update
    old_mean = ct.astype(ct.load(running_mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    old_var = ct.astype(ct.load(running_var_ptr, index=(channel,), shape=(1,)), ct.float32)
    new_running_mean = old_mean * OLD_WEIGHT + mean * MOMENTUM
    new_running_var = old_var * OLD_WEIGHT + var_val * RUNNING_VAR_CORRECTION * MOMENTUM
    ct.store(running_mean_ptr, index=(channel,),
              tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + new_running_mean, (1,)))
    ct.store(running_var_ptr, index=(channel,),
              tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + new_running_var, (1,)))


@ct.kernel
def _center_pool_kernel(
    x0_ptr,        # f32 flat NHWC
    x1_ptr,        # bf16 flat NHWC
    mean_ptr,      # f32 [C]
    invstd_ptr,    # f32 [C]
    weight_ptr,    # f32 [C]
    bias_ptr,      # f32 [C]
    pooled_out_ptr, # bf16 [N * C] flat
    centered_out_ptr,  # f32 flat NHWC
    BLOCK_X: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    x_pid = ct.bid(0)
    x_offsets = x_pid * BLOCK_X + ct.arange(BLOCK_X, dtype=ct.int32)  # (BLOCK_X,)
    x_mask = x_offsets < (N * C)  # per (n, c) pair
    channel = x_offsets - (x_offsets // C) * C
    batch = x_offsets // C

    # 2D tile: (BLOCK_X, BLOCK_R) where BLOCK_R covers HW.
    r_1d = ct.arange(BLOCK_R, dtype=ct.int32)
    r_mask = r_1d < HW
    zero_bool_2d = ct.full((BLOCK_X, BLOCK_R), False, dtype=ct.bool_)
    x_mask_2d = ct.reshape(x_mask, (BLOCK_X, 1)) | zero_bool_2d
    r_mask_2d = ct.reshape(r_mask, (1, BLOCK_R)) | zero_bool_2d
    mask = x_mask_2d & r_mask_2d
    zero_i32_2d = ct.zeros((BLOCK_X, BLOCK_R), dtype=ct.int32)
    channel_2d = ct.reshape(channel, (BLOCK_X, 1)) + zero_i32_2d
    batch_2d = ct.reshape(batch, (BLOCK_X, 1)) + zero_i32_2d
    r_2d = ct.reshape(r_1d, (1, BLOCK_R)) + zero_i32_2d

    # Flat NHWC offset: n * HW * C + hw * C + c
    offsets = batch_2d * (HW * C) + r_2d * C + channel_2d

    x0 = ct.gather(x0_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0))
    x1 = ct.astype(ct.gather(x1_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)), ct.float32)
    added = x0 + x1

    channel_1d_for_scalar = ct.reshape(channel, (BLOCK_X,))
    mean = ct.gather(mean_ptr, (channel_1d_for_scalar,), mask=x_mask, padding_value=ct.float32(0.0))
    invstd = ct.gather(invstd_ptr, (channel_1d_for_scalar,), mask=x_mask, padding_value=ct.float32(0.0))
    weight = ct.gather(weight_ptr, (channel_1d_for_scalar,), mask=x_mask, padding_value=ct.float32(0.0))
    bias = ct.gather(bias_ptr, (channel_1d_for_scalar,), mask=x_mask, padding_value=ct.float32(0.0))
    mean_2d = ct.reshape(mean, (BLOCK_X, 1))
    invstd_2d = ct.reshape(invstd, (BLOCK_X, 1))
    weight_2d = ct.reshape(weight, (BLOCK_X, 1))
    bias_2d = ct.reshape(bias, (BLOCK_X, 1))

    centered = added - mean_2d
    normalized = centered * invstd_2d
    affine = normalized * weight_2d + bias_2d
    zero_2d = ct.zeros((BLOCK_X, BLOCK_R), dtype=ct.float32)
    affine_masked = ct.where(mask, affine, zero_2d)
    affine_sum = ct.sum(affine_masked, axis=1)
    pooled = affine_sum / float(HW)
    pooled_bf16 = ct.astype(pooled, ct.bfloat16)
    ct.scatter(pooled_out_ptr, (x_offsets,), pooled_bf16, mask=x_mask)

    ct.scatter(centered_out_ptr, (offsets,), centered, mask=mask)


@oracle_impl(
    hardware="B200",
    point="d9d8b8eb",
    STATS_BLOCK_X=16,
    STATS_BLOCK_R=128,
    FINAL_BLOCK_C=32,
    FINAL_BLOCK_R=64,
    CENTER_BLOCK_X=128,
    CENTER_BLOCK_R=64,
)
def oracle_forward(
    inputs,
    *,
    STATS_BLOCK_X: int,
    STATS_BLOCK_R: int,
    FINAL_BLOCK_C: int,
    FINAL_BLOCK_R: int,
    CENTER_BLOCK_X: int,
    CENTER_BLOCK_R: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    device = arg0_1.device
    mean = torch.empty((C,), device=device, dtype=torch.float32)
    invstd = torch.empty((C,), device=device, dtype=torch.float32)
    pooled = torch.empty_strided((N, C), (C, 1), device=device, dtype=torch.bfloat16)
    # centered has channels-last stride (37632, 1, 5376, 768)
    centered_nhwc = torch.empty((N, H, W, C), device=device, dtype=torch.float32)
    centered = centered_nhwc.permute(0, 3, 1, 2)

    # Flatten channels-last inputs
    arg0_flat = arg0_1.permute(0, 2, 3, 1).reshape(-1)  # NHWC f32
    arg1_flat = arg1_1.permute(0, 2, 3, 1).reshape(-1)  # NHWC bf16

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_stats_kernel,
        (arg0_flat, arg1_flat, arg2_1, arg3_1, mean, invstd, 8192),
    )

    # Kernel 2: center + pool. BLOCK_X = number of (n, c) pairs per program.
    total_nc = N * C
    ct.launch(
        stream,
        ((total_nc + 128 - 1) // 128, 1, 1),
        _center_pool_kernel,
        (
            arg0_flat, arg1_flat,
            mean, invstd, arg4_1, arg5_1,
            pooled.view(-1),
            centered_nhwc.view(-1),
            128, 64,
        ),
    )

    return invstd, pooled, centered, arg2_1, arg3_1
