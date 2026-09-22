"""cuTile port of sum_sum_sum_59c5b1609d60: DenseNet BN-backward + residual add + sum-3.

Uses cuTile for the per-channel sum/dot reductions and final bf16 sum reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 32
H = 32
W = 32
HW = H * W
K_TOTAL = N * HW  # 128 * 1024 = 131072
REDUCE_SCALE = 7.62939453125e-06


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
def _channel_sum_bf16_reduce_kernel(
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
        acc_f = acc_f + ct.sum(x_masked)
    acc_bf = ct.astype(acc_f, ct.bfloat16)
    acc_f_out = ct.astype(acc_bf, ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(acc_f_out, (1,)))


@oracle_impl(hardware="B200", point="7ba0254f")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device

    slice_1 = arg0_1[:, 16:48, :, :]  # bf16 [128, 32, 32, 32]
    le = arg1_1 <= 0
    where_ = torch.where(le, arg2_1, arg3_1)
    where_f = where_.to(torch.float32)
    centered = arg4_1.to(torch.float32) - arg5_1

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

    mul_1 = sum_1 * REDUCE_SCALE
    unsqueeze_2 = mul_1.view(1, C, 1, 1)
    mul_2 = sum_2 * REDUCE_SCALE
    mul_3 = arg6_1 * arg6_1
    mul_4 = mul_2 * mul_3
    unsqueeze_5 = mul_4.view(1, C, 1, 1)
    mul_5 = arg6_1 * arg7_1
    unsqueeze_8 = mul_5.view(1, C, 1, 1)
    sub = arg4_1.to(torch.float32) - arg5_1
    mul_6 = sub * unsqueeze_5
    sub_1 = where_f - mul_6
    sub_2 = sub_1 - unsqueeze_2
    mul_7 = sub_2 * unsqueeze_8
    mul_8 = sum_2 * arg6_1
    dense = mul_7.to(torch.bfloat16)

    add_ = slice_1 + dense

    add_reshape = add_.permute(1, 0, 2, 3).contiguous().view(C, K_TOTAL)
    sum_3_f = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, 1, 1), _channel_sum_bf16_reduce_kernel,
        (add_reshape, sum_3_f, K_TOTAL, BLOCK_K, N_K_BLOCKS),
    )

    return sum_1, mul_8, add_, sum_3_f
