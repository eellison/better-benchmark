"""cuTile port of sum_sum_b1d8468e9c69: DenseNet BN-backward + residual slice-add.

Uses torch for pointwise/slice/where/BN-backward and a cuTile channel-reduction
kernel for the two per-channel sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


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


@oracle_impl(hardware="B200", point="f1255d51", H=14, W=14)
@oracle_impl(hardware="B200", point="6cbb3519", H=7, W=7)
def oracle_forward(inputs, *, H: int, W: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    n = int(arg1_1.shape[0])
    c = int(arg1_1.shape[1])
    device = arg0_1.device
    # HW_INV is the constant baked into the traced Repro graph (from the
    # 14x14 training shape, 1/784). Both shape points reuse the same constant.
    hw_inv = 0.0012755102040816326

    slice_1 = arg0_1[:, 960:992]
    le = arg1_1 <= 0
    where = torch.where(le, arg2_1, arg3_1)
    where_f = where.to(torch.float32)
    sub = arg4_1.to(torch.float32) - arg5_1

    where_c = where_f.permute(1, 0, 2, 3).contiguous().view(c, n * H * W)
    sub_c = sub.permute(1, 0, 2, 3).contiguous().view(c, n * H * W)
    sum_1 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((c,), device=device, dtype=torch.float32)
    BLOCK_K = _next_pow2(n * H * W)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1),
        _channel_reduce_kernel,
        (where_c, sub_c, sum_1, sum_2, n * H * W, BLOCK_K),
    )

    mul_1 = sum_1 * hw_inv
    mul_2 = sum_2 * hw_inv
    mul_3 = arg6_1 * arg6_1
    mul_4 = mul_2 * mul_3
    mul_5 = arg6_1 * arg7_1

    mul_6 = sub * mul_4.view(1, c, 1, 1)
    sub_1 = where_f - mul_6
    sub_2 = sub_1 - mul_1.view(1, c, 1, 1)
    mul_7 = sub_2 * mul_5.view(1, c, 1, 1)
    mul_8 = sum_2 * arg6_1
    conv2 = mul_7.to(torch.bfloat16)
    slice_2 = conv2[:, 960:992]
    add = slice_1 + slice_2

    return sum_1, mul_8, conv2, add
