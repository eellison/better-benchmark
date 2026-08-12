"""cuTile port of sum_sum_sum_b925d072e497: ShuffleNet channel-shuffle + dual BN-backward.

Channel shuffle done in torch:
- cat = [arg0[:, 0:116], arg1] along dim 1 -> [128, 232, ...]
- view as [128, 116, 2, 14, 14], permute to [128, 2, 116, 14, 14], flatten
- slice_2 = view_1[:, 0:116], slice_3 = view_1[:, 116:232]

Then for each branch: BN affine -> ReLU -> where(le, fill, slice) -> BN backward.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 3.985969387755102e-05


@ct.kernel
def _bn_relu_where_kernel(
    x_ptr, slice_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, fill_ptr, out_ptr,
    C_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    slice_val = ct.load(slice_ptr, index=(pid,), shape=(BLOCK,))
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    bias = ct.gather(bias_ptr, channel)

    x_f = ct.astype(x, ct.float32)
    centered = x_f - mean
    normed = centered * invstd
    affine = normed * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_r = ct.astype(affine_bf, ct.float32)
    relu = ct.where(affine_r > 0.0, affine_r, ct.full((BLOCK,), 0.0, dtype=ct.float32))
    le_zero = relu <= 0.0

    f_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bcast = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16) + ct.reshape(f_tile, (1,))
    result = ct.where(le_zero, fill_bcast, slice_val)
    ct.store(out_ptr, index=(pid,), tile=result)


@ct.kernel
def _channel_reduce_4outs_kernel(
    where1_ptr,   # bf16 [C, NHW]
    where2_ptr,   # bf16 [C, NHW]
    x1_ptr,       # bf16 [C, NHW]
    x2_ptr,       # bf16 [C, NHW]
    mean1_ptr,    # f32  [C]
    mean2_ptr,    # f32  [C]
    sum1_ptr,     # f32  [C]  <- where1.sum
    sum2_ptr,     # f32  [C]  <- (where1 * (x1 - mean1)).sum
    sum3_ptr,     # f32  [C]  <- where2.sum
    sum4_ptr,     # f32  [C]  <- (where2 * (x2 - mean2)).sum
    NHW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    N_TILES: ct.Constant[int],
):
    c_pid = ct.bid(0)

    mean1_tile = ct.load(mean1_ptr, index=(c_pid,), shape=(1,))
    mean2_tile = ct.load(mean2_ptr, index=(c_pid,), shape=(1,))
    mean1_2d = ct.reshape(mean1_tile, (1, 1))
    mean2_2d = ct.reshape(mean2_tile, (1, 1))

    acc1 = ct.zeros((1,), dtype=ct.float32)
    acc2 = ct.zeros((1,), dtype=ct.float32)
    acc3 = ct.zeros((1,), dtype=ct.float32)
    acc4 = ct.zeros((1,), dtype=ct.float32)

    for tile_idx in range(N_TILES):
        w1 = ct.load(where1_ptr, index=(c_pid, tile_idx), shape=(1, BLOCK_HW),
                     padding_mode=ct.PaddingMode.ZERO)
        x1 = ct.load(x1_ptr, index=(c_pid, tile_idx), shape=(1, BLOCK_HW),
                     padding_mode=ct.PaddingMode.ZERO)
        w2 = ct.load(where2_ptr, index=(c_pid, tile_idx), shape=(1, BLOCK_HW),
                     padding_mode=ct.PaddingMode.ZERO)
        x2 = ct.load(x2_ptr, index=(c_pid, tile_idx), shape=(1, BLOCK_HW),
                     padding_mode=ct.PaddingMode.ZERO)

        w1_f = ct.astype(w1, ct.float32)
        x1_f = ct.astype(x1, ct.float32)
        w2_f = ct.astype(w2, ct.float32)
        x2_f = ct.astype(x2, ct.float32)

        # Out-of-bounds loads are 0 due to ZERO padding, and w{1,2}_f is 0 at
        # those positions, so `w * (x - mean)` at invalid positions is 0.
        sub1 = x1_f - mean1_2d
        sub2 = x2_f - mean2_2d

        acc1 = acc1 + ct.sum(w1_f, axis=1)
        acc2 = acc2 + ct.sum(w1_f * sub1, axis=1)
        acc3 = acc3 + ct.sum(w2_f, axis=1)
        acc4 = acc4 + ct.sum(w2_f * sub2, axis=1)

    ct.store(sum1_ptr, index=(c_pid,), tile=acc1)
    ct.store(sum2_ptr, index=(c_pid,), tile=acc2)
    ct.store(sum3_ptr, index=(c_pid,), tile=acc3)
    ct.store(sum4_ptr, index=(c_pid,), tile=acc4)


@ct.kernel
def _bn_grad_kernel(
    where_ptr, x_ptr,
    mean_ptr, output_scale_ptr, var_term_ptr, mean_term_ptr,
    out_ptr,
    C_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    where_val = ct.load(where_ptr, index=(pid,), shape=(BLOCK,))
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    where_f = ct.astype(where_val, ct.float32)
    x_f = ct.astype(x, ct.float32)

    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    var_term = ct.gather(var_term_ptr, channel)
    mean_term = ct.gather(mean_term_ptr, channel)
    output_scale = ct.gather(output_scale_ptr, channel)

    centered = x_f - mean
    correction = centered * var_term
    after_corr = where_f - correction
    centered_grad = after_corr - mean_term
    grad = centered_grad * output_scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(grad, ct.bfloat16))


@oracle_impl(hardware="B200", point="8d158a8e", REDUCE_BLOCK=1024, BLOCK_C=4, EPILOGUE_BLOCK=512)
def oracle_forward(inputs, *, REDUCE_BLOCK: int, BLOCK_C: int, EPILOGUE_BLOCK: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, shape_view, shape_flat,
    ) = inputs
    n, input_c, h, w = (int(d) for d in arg0_1.shape)
    c = int(arg2_1.shape[1])
    device = arg0_1.device

    slice_1_full = arg0_1[:, :c, :, :]
    cat = torch.cat([slice_1_full, arg1_1], dim=1)
    view_shape = tuple(int(d) for d in shape_view)
    flat_shape = tuple(int(d) for d in shape_flat)
    view_r = cat.view(view_shape)
    permuted = view_r.permute(0, 2, 1, 3, 4).contiguous()
    view_1 = permuted.view(flat_shape)
    slice_2 = view_1[:, :c, :, :].contiguous()
    slice_3 = view_1[:, c:2 * c, :, :].contiguous()

    mean1 = arg3_1.view(c)
    invstd1 = arg4_1.view(c)
    weight1 = arg5_1.view(c)
    bias1 = arg6_1.view(c)
    fill = arg7_1.view(1)
    mean2 = arg9_1.view(c)
    invstd2 = arg10_1.view(c)
    weight2 = arg11_1.view(c)
    bias2 = arg12_1.view(c)

    arg2_nhwc = arg2_1.permute(0, 2, 3, 1).contiguous().view(-1)
    slice_3_nhwc = slice_3.permute(0, 2, 3, 1).contiguous().view(-1)
    arg8_nhwc = arg8_1.permute(0, 2, 3, 1).contiguous().view(-1)
    slice_2_nhwc = slice_2.permute(0, 2, 3, 1).contiguous().view(-1)

    n_flat = arg2_nhwc.numel()
    BLOCK = 512
    stream = torch.cuda.current_stream()

    where1_flat = torch.empty(n_flat, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _bn_relu_where_kernel,
        (arg2_nhwc, slice_3_nhwc, mean1, invstd1, weight1, bias1, fill, where1_flat,
         c, BLOCK),
    )

    where2_flat = torch.empty(n_flat, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _bn_relu_where_kernel,
        (arg8_nhwc, slice_2_nhwc, mean2, invstd2, weight2, bias2, fill, where2_flat,
         c, BLOCK),
    )

    # Permute to [C, NHW] so each cuTile program handles one channel with
    # contiguous per-channel loads. This mirrors Triton's channel-partitioned
    # partial reduction (`_partial_two_branch_kernel`).
    nhw = n * h * w
    where1_c = where1_flat.view(n, h, w, c).permute(3, 0, 1, 2).contiguous().view(c, nhw)
    where2_c = where2_flat.view(n, h, w, c).permute(3, 0, 1, 2).contiguous().view(c, nhw)
    x1_c = arg2_nhwc.view(n, h, w, c).permute(3, 0, 1, 2).contiguous().view(c, nhw)
    x2_c = arg8_nhwc.view(n, h, w, c).permute(3, 0, 1, 2).contiguous().view(c, nhw)

    BLOCK_HW = 1024
    n_tiles = ct.cdiv(nhw, BLOCK_HW)

    sum_1 = torch.empty(c, device=device, dtype=torch.float32)
    sum_2 = torch.empty(c, device=device, dtype=torch.float32)
    sum_3 = torch.empty(c, device=device, dtype=torch.float32)
    sum_4 = torch.empty(c, device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (c, 1, 1),
        _channel_reduce_4outs_kernel,
        (where1_c, where2_c, x1_c, x2_c, mean1, mean2,
         sum_1, sum_2, sum_3, sum_4,
         nhw, BLOCK_HW, n_tiles),
    )

    mul_10 = sum_2 * invstd1
    mul_21 = sum_4 * invstd2

    output_scale_1 = invstd1 * weight1
    var_term_1 = sum_2 * SCALE * invstd1 * invstd1
    mean_term_1 = sum_1 * SCALE

    out1_flat = torch.empty(n_flat, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _bn_grad_kernel,
        (where1_flat, arg2_nhwc, mean1, output_scale_1, var_term_1, mean_term_1, out1_flat,
         c, BLOCK),
    )
    grad1 = out1_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)

    output_scale_2 = invstd2 * weight2
    var_term_2 = sum_4 * SCALE * invstd2 * invstd2
    mean_term_2 = sum_3 * SCALE

    out2_flat = torch.empty(n_flat, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _bn_grad_kernel,
        (where2_flat, arg8_nhwc, mean2, output_scale_2, var_term_2, mean_term_2, out2_flat,
         c, BLOCK),
    )
    grad2 = out2_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)

    return sum_1, mul_10, grad1, sum_3, mul_21, grad2
