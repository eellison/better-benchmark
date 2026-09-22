"""cuTile port of sum_sum_sum_9ab0961d1fd9: UNet max-pool-backward + BN-backward.

Uses torch for the scatter-add + max-pool offset conversion + BN-backward
elementwise stages. cuTile provides the channel-reduction kernel for
sum_1 (= sum where_f) and sum_2 (= sum where_f * sub_1).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


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


@ct.kernel
def _channel_sum_bf16_kernel(
    x_ptr,          # bf16 [C, N*H*W]
    sum_out_ptr,    # f32 [C]
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


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _shape(shape):
    return tuple(int(dim) for dim in shape)


HW_INV = 0.0001050420168067227


@oracle_impl(hardware="B200", point="a380a4be")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     shape_full, shape_view, shape_ks, shape_input, shape_stride,
     shape_view_1, shape_view_2) = inputs
    device = arg0_1.device
    slice_1 = arg0_1[:, 0:512]
    full = torch.zeros(_shape(shape_full), device=device, dtype=torch.float32)
    view_view = arg1_1.view(_shape(shape_view))

    # Compute max-pool indices via torch prim.
    idx = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg2_1, _shape(shape_ks), _shape(shape_input), _shape(shape_stride),
        [0, 0], [1, 1],
    )
    view_1 = idx.view(_shape(shape_view_1))
    conv0 = view_view.to(torch.float32)
    scatter_add = torch.scatter_add(full, 1, view_1, conv0)
    view_2 = scatter_add.view(_shape(shape_view_2))
    conv1 = view_2.to(torch.bfloat16)
    add = slice_1 + conv1

    sub = arg3_1.to(torch.float32) - arg4_1
    mul = sub * arg5_1
    mul_1 = mul * arg6_1.view(512, 1, 1)
    add_1 = mul_1 + arg7_1.view(512, 1, 1)
    conv2 = add_1.to(torch.bfloat16)
    relu = torch.relu(conv2)
    le = relu <= 0
    where = torch.where(le, arg8_1, add)
    where_f = where.to(torch.float32)

    # Channel reductions.
    c = 512
    n, h, w = 1, 80, 119
    n_elem = n * h * w
    conv_arg3 = arg3_1.to(torch.float32)
    sub_1 = conv_arg3 - arg4_1

    where_c = where_f.permute(1, 0, 2, 3).contiguous().view(c, n_elem)
    sub_1_c = sub_1.permute(1, 0, 2, 3).contiguous().view(c, n_elem)
    sum_1 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((c,), device=device, dtype=torch.float32)
    BLOCK_K = _next_pow2(n_elem)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1),
        _channel_reduce_kernel,
        (where_c, sub_1_c, sum_1, sum_2, n_elem, BLOCK_K),
    )

    mul_3 = sum_1 * HW_INV
    mul_4 = sum_2 * HW_INV
    squeeze_1 = arg5_1.view(c)
    mul_5 = squeeze_1 * squeeze_1
    mul_6 = mul_4 * mul_5
    mul_7 = squeeze_1 * arg6_1

    mul_8 = sub_1 * mul_6.view(1, c, 1, 1)
    sub_2 = where_f - mul_8
    sub_3 = sub_2 - mul_3.view(1, c, 1, 1)
    mul_9 = sub_3 * mul_7.view(1, c, 1, 1)
    mul_10 = sum_2 * squeeze_1
    conv5 = mul_9.to(torch.bfloat16)

    # sum_3 = sum(conv5 over N/H/W) per C, with bf16 accumulator rounding to
    # match Triton's `_final_output_kernel` (`.to(bf16).to(f32)`).
    conv5_c = conv5.permute(1, 0, 2, 3).contiguous().view(c, n_elem)
    sum_3_f = torch.empty((c,), device=device, dtype=torch.float32)
    BLOCK_K_SUM3 = 4096
    N_K_BLOCKS_SUM3 = (n_elem + BLOCK_K_SUM3 - 1) // BLOCK_K_SUM3
    ct.launch(
        stream, (c, 1, 1), _channel_sum_bf16_kernel,
        (conv5_c, sum_3_f, n_elem, BLOCK_K_SUM3, N_K_BLOCKS_SUM3),
    )

    return sum_1, mul_10, conv5, sum_3_f
