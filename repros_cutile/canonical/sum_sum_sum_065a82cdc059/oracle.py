"""cuTile port of sum_sum_sum_065a82cdc059: RepVGG BN-backward triple pipeline.

Elementwise chain done with torch; cuTile channel reduction kernel handles
the four per-channel sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


# Constant from traced graph — always 1 / (128 * 56 * 56).
HW_INV = 2.4912308673469386e-06


@ct.kernel
def _channel_reduce_kernel(
    x_ptr,          # f32 (c, e)
    y_ptr,          # f32 (c, e)
    sum_x_ptr,      # f32 (c,)
    sum_xy_ptr,     # f32 (c,)
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


def _bn_reduce(x_f, sub, c, n_elem, device):
    """Compute (sum(x), sum(x * sub)) per-channel using cuTile."""
    x_c = x_f.permute(1, 0, 2, 3).contiguous().view(c, n_elem)
    sub_c = sub.permute(1, 0, 2, 3).contiguous().view(c, n_elem)
    sum_x = torch.empty((c,), device=device, dtype=torch.float32)
    sum_xy = torch.empty((c,), device=device, dtype=torch.float32)
    BLOCK_K = _next_pow2(n_elem)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1),
        _channel_reduce_kernel,
        (x_c, sub_c, sum_x, sum_xy, n_elem, BLOCK_K),
    )
    return sum_x, sum_xy


def _bn_backward(x_f, sub, sum_x, sum_xy, weight, bias, c):
    mul_1 = sum_x * HW_INV
    mul_2 = sum_xy * HW_INV
    mul_3 = weight * weight
    mul_4 = mul_2 * mul_3
    mul_5 = weight * bias
    mul_6 = sub * mul_4.view(1, c, 1, 1)
    sub_1 = x_f - mul_6
    sub_2 = sub_1 - mul_1.view(1, c, 1, 1)
    mul_7 = sub_2 * mul_5.view(1, c, 1, 1)
    return mul_7.to(torch.bfloat16)


@oracle_impl(hardware="B200", point="f665655f")
@oracle_impl(hardware="B200", point="ff6d7f0c")
@oracle_impl(hardware="B200", point="c0fba172")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, arg11_1,
     arg12_1, arg13_1, arg14_1, arg15_1) = inputs
    n = int(arg2_1.shape[0])
    c = int(arg2_1.shape[1])
    h = int(arg2_1.shape[2])
    w = int(arg2_1.shape[3])
    n_elem = n * h * w
    device = arg2_1.device

    add = arg0_1 + arg1_1

    # First BN backward using arg2_1 as gradient and arg3_1 as input.
    sub = arg3_1.to(torch.float32) - arg4_1
    sum_1, sum_2 = _bn_reduce(arg2_1, sub, c, n_elem, device)
    mul_7_bf = _bn_backward(arg2_1, sub, sum_1, sum_2, arg5_1, arg6_1, c)
    mul_8 = sum_2 * arg5_1
    add_1 = add + mul_7_bf

    # ReLU-backward-like gate: where(arg3_1 <= 0, arg7_1, add_1)
    le = arg3_1 <= 0
    where = torch.where(le, arg7_1, add_1)
    where_f = where.to(torch.float32)

    # Second BN backward using where_f as gradient, arg8_1 as input.
    sub_3 = arg8_1.to(torch.float32) - arg9_1
    sum_3, sum_4 = _bn_reduce(where_f, sub_3, c, n_elem, device)
    mul_16_bf = _bn_backward(where_f, sub_3, sum_3, sum_4, arg10_1, arg11_1, c)
    mul_17 = sum_4 * arg10_1

    # Third BN backward using where_f as gradient, arg12_1 as input.
    sub_6 = arg12_1.to(torch.float32) - arg13_1
    sum_5, sum_6 = _bn_reduce(where_f, sub_6, c, n_elem, device)
    mul_25_bf = _bn_backward(where_f, sub_6, sum_5, sum_6, arg14_1, arg15_1, c)
    mul_26 = sum_6 * arg14_1

    return sum_1, mul_8, sum_3, mul_17, mul_16_bf, sum_5, mul_26, mul_25_bf
