"""cuTile port of sum_sum_b67c187f8e8e: GhostNet BN-backward with masked producer.

Uses cuTile for the per-channel sum/dot reductions, then torch for the pointwise
BN-backward math.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 100
H = 14
W = 14
HW = H * W
K_TOTAL = N * HW
REDUCE_SCALE = 9.964923469387754e-06


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


@oracle_impl(hardware="B200", point="9343a6ce")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1) = inputs
    device = arg0_1.device

    # Step 1: build where = where(le(arg2_1, 0), arg3_1, arg0_1[:,:100] + arg1_1)
    slice_1 = arg0_1[:, :100, :, :]
    add_ = slice_1 + arg1_1
    le = arg2_1 <= 0
    where_ = torch.where(le, arg3_1, add_)  # bf16 [N, 100, H, W]
    where_f = where_.to(torch.float32)

    # Step 2: centered = arg4_1.to(f32) - arg5_1
    centered = arg4_1.to(torch.float32) - arg5_1

    # cuTile reduction: per-C sum and dot(where_f, centered) over N/H/W.
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

    # BN-backward math via torch.
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
    dense_out = mul_7.to(torch.bfloat16)

    return sum_1, mul_8, dense_out
