"""cuTile port of sum_sum_sum_0db891d3a1f0: UNet BN-backward + sum-3 reduction.

Uses cuTile for the two per-channel sum/dot reductions and the final third
per-channel bf16 sum. Everything else is torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 1
C = 64
H = 640
W = 959
HW = H * W
K_TOTAL = N * HW  # 640 * 959 = 613760
REDUCE_SCALE = 1.6293013555787278e-06


@ct.kernel
def _channel_sum_dot_kernel(
    where_ptr,           # f32 [C, K_TOTAL]
    centered_ptr,        # f32 [C, K_TOTAL]
    sum_out_ptr,         # f32 [C]
    dot_out_ptr,         # f32 [C]
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    N_K_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    sum_acc = ct.zeros((), dtype=ct.float32)
    dot_acc = ct.zeros((), dtype=ct.float32)
    for kb in range(N_K_BLOCKS):
        where_f = ct.load(
            where_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        centered_f = ct.load(
            centered_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ks = ct.arange(BLOCK_K, dtype=ct.int32) + kb * BLOCK_K
        valid = ks < K_TOTAL_
        valid_2d = ct.reshape(valid, (1, BLOCK_K))
        zero_f = ct.full((1, BLOCK_K), 0.0, dtype=ct.float32)
        where_masked = ct.where(valid_2d, where_f, zero_f)
        centered_masked = ct.where(valid_2d, centered_f, zero_f)
        sum_acc = sum_acc + ct.sum(where_masked)
        dot_acc = dot_acc + ct.sum(where_masked * centered_masked)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_acc, (1,)))
    ct.store(dot_out_ptr, index=(c,), tile=ct.reshape(dot_acc, (1,)))


@ct.kernel
def _channel_sum_bf16_kernel(
    x_ptr,               # bf16 [C, K_TOTAL]
    sum_out_ptr,         # f32 [C]
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    N_K_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    acc_f = ct.zeros((), dtype=ct.float32)
    for kb in range(N_K_BLOCKS):
        x_bf = ct.load(
            x_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ks = ct.arange(BLOCK_K, dtype=ct.int32) + kb * BLOCK_K
        valid = ks < K_TOTAL_
        valid_2d = ct.reshape(valid, (1, BLOCK_K))
        zero_f = ct.full((1, BLOCK_K), 0.0, dtype=ct.float32)
        x_f = ct.astype(x_bf, ct.float32)
        x_masked = ct.where(valid_2d, x_f, zero_f)
        block_sum = ct.sum(x_masked)
        acc_f = acc_f + block_sum
    # Round to bf16 then back to f32 to match Triton's `.to(bf16).to(f32)`.
    acc_bf = ct.astype(acc_f, ct.bfloat16)
    acc_f_out = ct.astype(acc_bf, ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(acc_f_out, (1,)))


@oracle_impl(hardware="B200", point="0884bda3")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    le = arg0_1 <= 0
    full_scalar = torch.zeros((), device=device, dtype=torch.bfloat16)
    where_ = torch.where(le, full_scalar, arg1_1)
    where_f = where_.to(torch.float32)
    centered = arg2_1.to(torch.float32) - arg3_1

    # Reductions.
    where_reshape = where_f.permute(1, 0, 2, 3).contiguous().view(C, K_TOTAL)
    centered_reshape = centered.permute(1, 0, 2, 3).contiguous().view(C, K_TOTAL)

    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C,), device=device, dtype=torch.float32)

    BLOCK_K = 4096
    N_K_BLOCKS = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _channel_sum_dot_kernel,
        (where_reshape, centered_reshape, sum_1, sum_2,
         K_TOTAL, BLOCK_K, N_K_BLOCKS),
    )

    # BN-backward.
    mul_1 = sum_1 * REDUCE_SCALE
    unsqueeze_2 = mul_1.view(1, C, 1, 1)
    mul_2 = sum_2 * REDUCE_SCALE
    mul_3 = arg4_1 * arg4_1
    mul_4 = mul_2 * mul_3
    unsqueeze_5 = mul_4.view(1, C, 1, 1)
    mul_5 = arg4_1 * arg5_1
    unsqueeze_8 = mul_5.view(1, C, 1, 1)
    sub = arg2_1.to(torch.float32) - arg3_1
    mul_6 = sub * unsqueeze_5
    sub_1 = where_f - mul_6
    sub_2 = sub_1 - unsqueeze_2
    mul_7 = sub_2 * unsqueeze_8
    mul_8 = sum_2 * arg4_1
    dense_out = mul_7.to(torch.bfloat16)

    # sum_3 = sum(dense_out over N/H/W) per C, in bf16 accumulation.
    dense_reshape = dense_out.permute(1, 0, 2, 3).contiguous().view(C, K_TOTAL)
    sum_3_f = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, 1, 1), _channel_sum_bf16_kernel,
        (dense_reshape, sum_3_f, K_TOTAL, BLOCK_K, N_K_BLOCKS),
    )

    return full_scalar, sum_1, mul_8, dense_out, sum_3_f
