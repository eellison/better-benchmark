"""cuTile port of sum_sum_792cabdf2c7c: DenseNet BN-backward + residual add.

Uses cuTile for the per-channel sum/dot reductions, then torch for the rest of
the pointwise BN-backward math and residual add.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 128
H = 56
W = 56
HW = H * W
K_TOTAL = N * HW
SLICE_START = 96
SLICE_C = 32
SCALE = 7.971938775510203e-05


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _channel_sum_dot_kernel(
    where_ptr,           # bf16 [C, K_TOTAL]
    centered_ptr,        # f32 [C, K_TOTAL]
    sum_out_ptr,         # f32 [C]
    dot_out_ptr,         # f32 [C]
    K_TOTAL_: ct.Constant[int],
    K_PAD: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    N_K_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    sum_acc = ct.zeros((), dtype=ct.float32)
    dot_acc = ct.zeros((), dtype=ct.float32)
    for kb in range(N_K_BLOCKS):
        where_bf = ct.load(
            where_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        centered_f = ct.load(
            centered_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        where_f = ct.astype(where_bf, ct.float32)
        # Mask OOB positions
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


@oracle_impl(hardware="B200", point="c9380b3e")
def oracle_forward(inputs):
    (r0, r1, r2, r3, mask, fill, source, centered_source, mean,
     invstd, weight) = inputs
    device = source.device

    # Step 1: compute where = where(mask <= 0, fill, source) as bf16.
    le = mask <= 0
    where_ = torch.where(le, fill, source)  # bf16 [N, C, H, W]
    where_f = where_.to(torch.float32)
    # Step 2: compute centered = centered_source - mean as f32.
    centered = centered_source.to(torch.float32) - mean

    # Step 3: sum_1 = sum(where_f over N/H/W per C) using cuTile.
    # Step 4: sum_2 = sum(where_f * centered) per C using cuTile.
    # Prepare [C, K_TOTAL] contiguous views.
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
         K_TOTAL, BLOCK_K * N_K_BLOCKS, BLOCK_K, N_K_BLOCKS),
    )

    # Rest of the BN backward computation in torch.
    mul_1 = sum_1 * SCALE
    unsqueeze_2 = mul_1.view(1, C, 1, 1)
    mul_2 = sum_2 * SCALE
    mul_3 = invstd * invstd
    mul_4 = mul_2 * mul_3
    unsqueeze_5 = mul_4.view(1, C, 1, 1)
    mul_5 = invstd * weight
    unsqueeze_8 = mul_5.view(1, C, 1, 1)
    sub = centered_source.to(torch.float32) - mean
    mul_6 = sub * unsqueeze_5
    sub_1 = where_f - mul_6
    sub_2 = sub_1 - unsqueeze_2
    mul_7 = sub_2 * unsqueeze_8
    mul_8 = sum_2 * invstd
    dense_out = mul_7.to(torch.bfloat16)

    # Add residual slice.
    slice_1 = r0[:, 96:128, :, :]
    slice_2 = r1[:, 96:128, :, :]
    slice_3 = r2[:, 96:128, :, :]
    slice_4 = r3[:, 96:128, :, :]
    add_ = slice_1 + slice_2
    add_ = add_ + slice_3
    add_ = add_ + slice_4
    slice_5 = dense_out[:, 96:128, :, :]
    add_out = add_ + slice_5

    return sum_1, mul_8, dense_out, add_out
