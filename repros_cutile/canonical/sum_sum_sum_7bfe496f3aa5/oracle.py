"""cuTile port of sum_sum_sum_7bfe496f3aa5: inception_v3 BN backward (4 branches).

Uses one cuTile elementwise kernel for the BN backward final step, invoked
once per branch. Reductions and reshapes are done in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
H = 35
W = 35
INV_N = 6.3775510204081635e-06  # 1/(128*35*35)
BLOCK = 1024


@ct.kernel
def _bn_backward_elem_kernel(
    conv_1_ptr,      # f32 [PIXELS]
    sub_1_ptr,       # f32 [PIXELS]
    scale_ptr,       # f32 [C]
    factor_ptr,      # f32 [C]
    var_scale_ptr,   # f32 [C]
    out_ptr,         # bf16 [PIXELS]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = ct.arange(BLOCK_, dtype=ct.int64) + pid * BLOCK_
    conv_1 = ct.load(conv_1_ptr, index=(pid,), shape=(BLOCK_,))
    sub_1 = ct.load(sub_1_ptr, index=(pid,), shape=(BLOCK_,))
    c = (idx // HW) % C
    scale = ct.gather(scale_ptr, c)
    factor = ct.gather(factor_ptr, c)
    var_scale = ct.gather(var_scale_ptr, c)
    mul_8 = sub_1 * var_scale
    sub_2 = conv_1 - mul_8
    sub_3 = sub_2 - factor
    mul_9 = sub_3 * scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(mul_9, ct.bfloat16))


def _bn_branch(x_input, slice_in, mean_arg, invstd_arg, weight_arg, bias_arg,
               fill_scalar, c, device):
    """One BN backward branch. Returns (sum_1, mul_10, out_bf16)."""
    # Forward BN
    sub = x_input.to(torch.float32) - mean_arg
    mul = sub * invstd_arg
    mul_1 = mul * weight_arg.view(1, c, 1, 1)
    add = mul_1 + bias_arg.view(1, c, 1, 1)
    conv = add.to(torch.bfloat16)
    relu = torch.relu(conv)
    le = relu <= 0
    where = torch.where(le, fill_scalar, slice_in)
    conv_1 = where.to(torch.float32)

    squeeze = mean_arg.squeeze(0).squeeze(-1).squeeze(-1)
    sum_1 = conv_1.sum(dim=(0, 2, 3))
    conv_2 = x_input.to(torch.float32)
    sub_1 = conv_2 - squeeze.view(1, c, 1, 1)
    mul_2 = conv_1 * sub_1
    sum_2 = mul_2.sum(dim=(0, 2, 3))

    mul_3 = sum_1 * INV_N
    mul_4 = sum_2 * INV_N
    squeeze_1 = invstd_arg.squeeze(0).squeeze(-1).squeeze(-1)
    mul_5 = squeeze_1 * squeeze_1
    mul_6 = mul_4 * mul_5  # var_scale
    mul_7 = squeeze_1 * weight_arg  # scale
    mul_10 = sum_2 * squeeze_1

    pixels = BATCH * c * H * W
    conv_1_flat = conv_1.contiguous().view(-1)
    sub_1_flat = sub_1.contiguous().view(-1)
    out = torch.empty_strided(
        (BATCH, c, H, W), (c * H * W, H * W, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(pixels, BLOCK), 1, 1), _bn_backward_elem_kernel,
        (conv_1_flat, sub_1_flat, mul_7.contiguous(),
         mul_3.contiguous(), mul_6.contiguous(), out_flat,
         c, H * W, BLOCK),
    )
    return sum_1, mul_10, out


@oracle_impl(hardware="B200", point="32cbe7ad")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
     arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1,
     arg24_1, arg25_1) = inputs
    device = arg0_1.device

    # Avg pool2d backward + add residuals
    avgpool_bw = torch.ops.aten.avg_pool2d_backward.default(
        arg0_1, arg1_1, [3, 3], [1, 1], [1, 1], False, True, None,
    )
    add = avgpool_bw + arg2_1
    add_1 = add + arg3_1
    add_2 = add_1 + arg4_1  # bf16 [B, 256, H, W]

    # Slice into 4 branches
    slice_1 = add_2[:, 0:64, :, :]
    slice_2 = add_2[:, 64:128, :, :]
    slice_3 = add_2[:, 128:224, :, :]
    slice_4 = add_2[:, 224:256, :, :]

    # Branch 1: C=32
    sum_1, mul_10, out_1 = _bn_branch(
        arg5_1, slice_4, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, 32, device,
    )
    # Branch 2: C=96
    sum_3, mul_21, out_2 = _bn_branch(
        arg11_1, slice_3, arg12_1, arg13_1, arg14_1, arg15_1, arg10_1, 96, device,
    )
    # Branch 3: C=64
    sum_5, mul_32, out_3 = _bn_branch(
        arg16_1, slice_2, arg17_1, arg18_1, arg19_1, arg20_1, arg10_1, 64, device,
    )
    # Branch 4: C=64
    sum_7, mul_43, out_4 = _bn_branch(
        arg21_1, slice_1, arg22_1, arg23_1, arg24_1, arg25_1, arg10_1, 64, device,
    )

    return (sum_1, mul_10, out_1, sum_3, mul_21, out_2,
            sum_5, mul_32, out_3, sum_7, mul_43, out_4)
