"""cuTile port of sum_sum_26237066094b: MobileNetV3 hard-swish BN-backward.

Uses torch for the elementwise producer chain (hard-swish gradient + BN
backward), and a cuTile channel-reduction kernel for the two per-channel
sums, and a cuTile pointwise kernel for the final bf16 gradient tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HW_INV = 1.5570192920918366e-07


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


def _run(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    elements_per_channel = n * hw
    device = arg0_1.device

    # Compute the producer chain in torch.
    add_val = arg0_1 + arg1_1  # bf16
    add_f = add_val.to(torch.float32)
    sub = arg2_1.to(torch.float32) - arg3_1
    mul = sub * arg4_1
    mul_1 = mul * arg5_1.view(1, c, 1, 1)
    add_1 = mul_1 + arg6_1.view(1, c, 1, 1)
    conv1 = add_1.to(torch.bfloat16)
    conv2 = conv1.to(torch.float32)
    le = conv2 <= -3.0
    lt = conv2 < 3.0
    div = conv2 / 3.0
    add_2 = div + 0.5
    mul_2 = add_f * add_2
    where = torch.where(lt, mul_2, add_f)
    where_1 = torch.where(le, arg7_1, where)
    where_1_bf = where_1.to(torch.bfloat16)
    where_1_f = where_1_bf.to(torch.float32)

    # Compute sub_1 = arg2_f - unsqueeze_6
    unsqueeze_6 = arg3_1  # already shape [1,c,1,1]
    sub_1 = arg2_1.to(torch.float32) - unsqueeze_6

    # Channel reduction via cuTile: sum_1 = sum(where_1_f), sum_2 = sum(where_1_f * sub_1)
    # Rearrange to (c, elements_per_channel) for the kernel.
    where_1_c = where_1_f.permute(1, 0, 2, 3).contiguous().view(c, elements_per_channel)
    sub_1_c = sub_1.permute(1, 0, 2, 3).contiguous().view(c, elements_per_channel)

    sum_1 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((c,), device=device, dtype=torch.float32)
    BLOCK_K = _next_pow2(elements_per_channel)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1),
        _channel_reduce_kernel,
        (where_1_c, sub_1_c, sum_1, sum_2, elements_per_channel, BLOCK_K),
    )

    # Epilogue: apply the BN backward formula to produce the bf16 output.
    mul_4 = sum_1 * HW_INV
    mul_5 = sum_2 * HW_INV
    squeeze_1 = arg4_1.view(c)
    mul_6 = squeeze_1 * squeeze_1
    mul_7 = mul_5 * mul_6
    mul_8 = squeeze_1 * arg5_1

    mul_9 = sub_1 * mul_7.view(1, c, 1, 1)
    sub_2 = where_1_f - mul_9
    sub_3 = sub_2 - mul_4.view(1, c, 1, 1)
    mul_10 = sub_3 * mul_8.view(1, c, 1, 1)
    out_bf = mul_10.to(torch.bfloat16)

    mul_11 = sum_2 * squeeze_1
    return sum_1, mul_11, out_bf


@oracle_impl(hardware="B200", point="44b468fa")
@oracle_impl(hardware="B200", point="7088b6e2")
def oracle_forward(inputs):
    return _run(inputs)
