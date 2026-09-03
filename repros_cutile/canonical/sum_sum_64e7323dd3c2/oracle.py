"""cuTile port of sum_sum_64e7323dd3c2: DenseNet BN-backward + residual slice-add.

Combines torch pointwise/slice ops for the sequential bf16 slice-add chain
and BN-backward epilogue, with a cuTile channel-reduction kernel for the
two per-channel sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HW_INV = 0.00031887755102040814  # 1 / (N * H * W) = 1/(4*28*28)


@ct.kernel
def _channel_reduce_kernel(
    x_ptr,          # f32 (channels, elements_per_channel)
    y_ptr,          # f32 (channels, elements_per_channel)
    sum_x_ptr,      # f32 (channels,)
    sum_xy_ptr,     # f32 (channels,)
    ELEMENTS_PER_CHANNEL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    channel = ct.bid(0)
    x = ct.load(x_ptr, index=(channel, 0), shape=(1, BLOCK_K),
                 padding_mode=ct.PaddingMode.ZERO)
    y = ct.load(y_ptr, index=(channel, 0), shape=(1, BLOCK_K),
                 padding_mode=ct.PaddingMode.ZERO)
    cols = ct.arange(BLOCK_K, dtype=ct.int32)
    col_valid = ct.reshape(cols < ELEMENTS_PER_CHANNEL, (1, BLOCK_K))
    zero_f = ct.zeros((1, BLOCK_K), dtype=ct.float32)
    x_m = ct.where(col_valid, x, zero_f)
    y_m = ct.where(col_valid, y, zero_f)
    sum_x = ct.sum(x_m)
    sum_xy = ct.sum(x_m * y_m)
    ct.store(sum_x_ptr, index=(channel,), tile=ct.reshape(sum_x, (1,)))
    ct.store(sum_xy_ptr, index=(channel,), tile=ct.reshape(sum_xy, (1,)))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="342bbb54")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1) = inputs
    n, c, h, w = 4, 384, 28, 28
    device = arg0_1.device

    # Sequential bf16 slice-add of the residual channels.
    slice_1 = arg0_1[:, 352:384]
    slice_2 = arg1_1[:, 352:384]
    add = slice_1 + slice_2  # bf16 rounding boundary each step
    slice_3 = arg2_1[:, 352:384]
    add_1 = add + slice_3
    slice_4 = arg3_1[:, 352:384]
    add_2 = add_1 + slice_4

    # BN-backward pipeline.
    le = arg4_1 <= 0
    where = torch.where(le, arg5_1, arg6_1)  # bf16
    where_f = where.to(torch.float32)
    sub = arg7_1.to(torch.float32) - arg8_1

    # Reductions via cuTile.
    where_c = where_f.permute(1, 0, 2, 3).contiguous().view(c, n * h * w)
    sub_c = sub.permute(1, 0, 2, 3).contiguous().view(c, n * h * w)
    sum_1 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((c,), device=device, dtype=torch.float32)
    BLOCK_K = _next_pow2(n * h * w)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1),
        _channel_reduce_kernel,
        (where_c, sub_c, sum_1, sum_2, n * h * w, BLOCK_K),
    )

    # BN-backward epilogue.
    mul_1 = sum_1 * HW_INV
    mul_2 = sum_2 * HW_INV
    mul_3 = arg9_1 * arg9_1
    mul_4 = mul_2 * mul_3
    mul_5 = arg9_1 * arg10_1

    mul_6 = sub * mul_4.view(1, c, 1, 1)
    sub_1 = where_f - mul_6
    sub_2 = sub_1 - mul_1.view(1, c, 1, 1)
    mul_7 = sub_2 * mul_5.view(1, c, 1, 1)
    mul_8 = sum_2 * arg9_1
    conv2 = mul_7.to(torch.bfloat16)
    slice_5 = conv2[:, 352:384]
    add_3 = add_2 + slice_5

    return sum_1, mul_8, conv2, add_3
