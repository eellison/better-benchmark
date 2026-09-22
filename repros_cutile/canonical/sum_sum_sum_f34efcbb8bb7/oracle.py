"""cuTile port of sum_sum_sum_f34efcbb8bb7: Adv-Inception avg-pool-backward + 6-branch BN-backward.

Uses two cuTile kernels per branch (BN partial + epilogue), invoked 6 times.
avg_pool_backward and slicing via torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 0.0001220703125


@ct.kernel
def _bn_partial_kernel(
    slice_ptr,     # bf16 [ELEMENTS_PER_CHANNEL, C_PAD]
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    where_out_ptr,
    partial_sum_ptr,
    partial_prod_ptr,
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    slice_1 = ct.load(slice_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))

    sub_f = x_f - mean_2d
    mul_f = sub_f * invstd_2d * weight_2d + bias_2d
    add_bf = ct.astype(mul_f, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.bfloat16)
    relu_bf = ct.where(add_bf > zero_bf, add_bf, zero_bf)
    le = relu_bf <= zero_bf

    fill_2d = ct.reshape(fill, (1, 1))
    where_bf = ct.where(le, fill_2d, slice_1)
    where_f = ct.astype(where_bf, ct.float32)
    ct.store(where_out_ptr, index=(chunk, 0), tile=where_f)

    prod = where_f * sub_f
    partial_sum = ct.sum(where_f, axis=0)
    partial_prod = ct.sum(prod, axis=0)
    ct.store(partial_sum_ptr, index=(chunk, 0), tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_prod_ptr, index=(chunk, 0), tile=ct.reshape(partial_prod, (1, BLOCK_C)))


@ct.kernel
def _bn_epilogue_kernel(
    where_ptr,
    x_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    where_f = ct.load(where_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    prod_coeff = ct.load(prod_coeff_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, BLOCK_C))
    output_scale_2d = ct.reshape(output_scale, (1, BLOCK_C))
    sub_f = x_f - mean_2d
    correction = sub_f * prod_coeff_2d
    res = where_f - correction - mean_term_2d
    out_f = res * output_scale_2d
    ct.store(out_ptr, index=(chunk, 0), tile=ct.astype(out_f, ct.bfloat16))


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def _bn_branch(slice_t, x, mean_1x1, invstd_1x1, weight_c, bias_c, fill, BLOCK_R):
    device = slice_t.device
    N, C, H, W = slice_t.shape
    HW = H * W
    epc = N * HW
    C_PAD = _next_pow2(C)

    def _nhwc(t):
        return t.permute(0, 2, 3, 1).contiguous().view(epc, C)

    def _pad_c(t):
        if t.shape[-1] == C_PAD:
            return t
        out = torch.zeros(t.shape[:-1] + (C_PAD,), device=t.device, dtype=t.dtype)
        out[..., :t.shape[-1]] = t
        return out

    slice_flat = _nhwc(slice_t)
    x_flat = _nhwc(x)
    mean_c = mean_1x1.view(C)
    invstd_c = invstd_1x1.view(C)

    slice_pad = _pad_c(slice_flat)
    x_pad = _pad_c(x_flat)
    mean_pad = _pad_c(mean_c)
    invstd_pad = _pad_c(invstd_c)
    weight_pad = _pad_c(weight_c)
    bias_pad = _pad_c(bias_c)
    fill_1d = fill.reshape(1)

    num_chunks = (epc + BLOCK_R - 1) // BLOCK_R
    where_out = torch.zeros((epc, C_PAD), device=device, dtype=torch.float32)
    partial_sum = torch.zeros((num_chunks, C_PAD), device=device, dtype=torch.float32)
    partial_prod = torch.zeros((num_chunks, C_PAD), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_chunks, 1, 1), _bn_partial_kernel,
        (slice_pad, x_pad, mean_pad, invstd_pad, weight_pad, bias_pad, fill_1d,
         where_out, partial_sum, partial_prod, BLOCK_R, C_PAD),
    )

    sum_1 = partial_sum[:, :C].sum(dim=0)
    prod_sum = partial_prod[:, :C].sum(dim=0)
    mul_10 = prod_sum * invstd_c
    mean_term = sum_1 * REDUCE_SCALE
    prod_coeff = (prod_sum * REDUCE_SCALE) * (invstd_c * invstd_c)
    output_scale = invstd_c * weight_c

    mean_term_pad = _pad_c(mean_term)
    prod_coeff_pad = _pad_c(prod_coeff)
    output_scale_pad = _pad_c(output_scale)

    out_flat_pad = torch.empty((epc, C_PAD), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream, (num_chunks, 1, 1), _bn_epilogue_kernel,
        (where_out, x_pad, mean_pad, mean_term_pad, prod_coeff_pad, output_scale_pad,
         out_flat_pad, BLOCK_R, C_PAD),
    )
    out_valid = out_flat_pad[:, :C].contiguous()
    out_nhwc = out_valid.view(N, H, W, C)
    out_shape = (N, C, H, W)
    out_stride = (C * HW, 1, C * W, C)
    out = torch.empty_strided(out_shape, out_stride, device=device, dtype=torch.bfloat16)
    out.copy_(out_nhwc.permute(0, 3, 1, 2))
    return sum_1, mul_10, out


@oracle_impl(hardware="B200", point="32065934", BLOCK_R=64)
def oracle_forward(inputs, *, BLOCK_R: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1,
     arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
     arg16_1, arg17_1, arg18_1, arg19_1, arg20_1,
     arg21_1, arg22_1, arg23_1, arg24_1, arg25_1,
     arg26_1, arg27_1, arg28_1, arg29_1, arg30_1,
     arg31_1, arg32_1, arg33_1, arg34_1, arg35_1) = inputs

    avg_pool_bwd = torch.ops.aten.avg_pool2d_backward.default(
        arg0_1, arg1_1, [3, 3], [1, 1], [1, 1], False, True, None,
    )
    add_2 = avg_pool_bwd + arg2_1 + arg3_1 + arg4_1

    slice_1 = add_2[:, 0:320]
    slice_2 = add_2[:, 320:1088]
    slice_3 = add_2[:, 1088:1856]
    slice_4 = add_2[:, 1856:2048]

    sum_1, mul_10, out_192 = _bn_branch(
        slice_4, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, BLOCK_R,
    )

    slice_5 = slice_3[:, 0:384]
    slice_6 = slice_3[:, 384:768]
    sum_3, mul_21, out_384_1 = _bn_branch(
        slice_6, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg10_1, BLOCK_R,
    )
    sum_5, mul_32, out_384_2 = _bn_branch(
        slice_5, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg10_1, BLOCK_R,
    )

    slice_7 = slice_2[:, 0:384]
    slice_8 = slice_2[:, 384:768]
    sum_7, mul_43, out_384_3 = _bn_branch(
        slice_8, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg10_1, BLOCK_R,
    )
    sum_9, mul_54, out_384_4 = _bn_branch(
        slice_7, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg10_1, BLOCK_R,
    )

    sum_11, mul_65, out_320 = _bn_branch(
        slice_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg10_1, BLOCK_R,
    )

    return (
        sum_1, mul_10, out_192,
        sum_3, mul_21, out_384_1,
        sum_5, mul_32, out_384_2,
        sum_7, mul_43, out_384_3,
        sum_9, mul_54, out_384_4,
        sum_11, mul_65, out_320,
    )
