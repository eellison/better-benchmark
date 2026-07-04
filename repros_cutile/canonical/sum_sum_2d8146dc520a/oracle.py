"""cuTile port of sum_sum_2d8146dc520a: Inception BN-backward with avgpool + maxpool scatter.

Uses cuTile for the per-channel sum/dot reductions. Everything else is torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 192
OUT_H = 71
OUT_W = 71
OUT_HW = OUT_H * OUT_W
N = 128
K_TOTAL = N * OUT_HW  # 128 * 5041 = 645248
REDUCTION_SCALE = 1.5497917079944455e-06


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


@oracle_impl(hardware="B200", point="e5f2e3e0")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, arg11_1,
     shape_p0, shape_p1, shape_p2, shape_p3, shape_p4, shape_p5, shape_p6) = inputs
    device = arg0_1.device
    kernel_size = list(shape_p2)
    input_size = list(shape_p3)
    stride_ = list(shape_p4)

    # avg_pool2d_backward + adds.
    avg_pool_back = torch.ops.aten.avg_pool2d_backward.default(
        arg0_1, arg1_1, [3, 3], [1, 1], [1, 1], False, True, None)
    add_ = avg_pool_back + arg2_1
    add_ = add_ + arg3_1
    add_ = add_ + arg4_1

    # Max-pool scatter.
    full = torch.zeros((24576, 5041), device=device, dtype=torch.float32)
    view = add_.contiguous().view(24576, 1225)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg5_1, kernel_size, input_size, stride_, [0, 0], [1, 1])
    indices_c = indices.contiguous().view(24576, 1225)
    scatter_add = torch.scatter_add(full, 1, indices_c, view.to(torch.float32))
    view_2 = scatter_add.view(128, 192, 71, 71)
    scatter_bf = view_2.to(torch.bfloat16)
    scatter_cl = scatter_bf.contiguous(memory_format=torch.channels_last)

    # BN affine + ReLU + where.
    sub = arg6_1 - arg7_1
    mul = sub * arg8_1
    mul_1 = mul * arg9_1.view(1, 192, 1, 1)
    add_3 = mul_1 + arg10_1.view(1, 192, 1, 1)
    add_3_bf = add_3.to(torch.bfloat16)
    relu = torch.relu(add_3_bf)
    le = relu <= 0
    where_ = torch.where(le, arg11_1, scatter_cl)
    where_f = where_.to(torch.float32)

    # Per-channel reductions.
    where_reshape = where_f.permute(1, 0, 2, 3).contiguous().view(C, K_TOTAL)
    # arg7_1 is [1, 192, 1, 1] f32. arg6_1 is [128, 192, 71, 71] bf16.
    centered = arg6_1.to(torch.float32) - arg7_1
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

    # Rest of BN-backward math.
    mul_3 = sum_1 * REDUCTION_SCALE
    unsqueeze_9 = mul_3.view(1, C, 1, 1)
    mul_4 = sum_2 * REDUCTION_SCALE
    squeeze_1 = arg8_1.squeeze()
    mul_5 = squeeze_1 * squeeze_1
    mul_6 = mul_4 * mul_5
    unsqueeze_12 = mul_6.view(1, C, 1, 1)
    mul_7 = squeeze_1 * arg9_1
    unsqueeze_15 = mul_7.view(1, C, 1, 1)
    sub_1 = arg6_1.to(torch.float32) - arg7_1
    mul_8 = sub_1 * unsqueeze_12
    sub_2 = where_f - mul_8
    sub_3 = sub_2 - unsqueeze_9
    mul_9 = sub_3 * unsqueeze_15
    mul_10 = sum_2 * squeeze_1
    dense_out = mul_9.to(torch.bfloat16)

    return sum_1, mul_10, dense_out
