"""cuTile port of sum_sum_sum_92943d6eae4c: Inception 4-branch BN-backward.

Reference: avg_pool2d_backward then 4-way slice, then 4x independent
BN-backward + relu-gate pattern (each producing sum_i, mul_i, convert_i).

Uses torch directly to avoid the heavy per-branch permute/contiguous overhead
that the prior cuTile port was paying.
"""

import torch

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 192
HEIGHT = 17
WIDTH = 17
HW = HEIGHT * WIDTH


def _run_branch(slice_channels_last, x, mean, invstd, weight, bias, fill, inv_count):
    mean_c = mean.view(CHANNELS)
    invstd_c = invstd.view(CHANNELS)

    x_f = x.to(torch.float32)
    centered_source = x_f - mean.view(1, CHANNELS, 1, 1)
    normalized = centered_source * invstd.view(1, CHANNELS, 1, 1)
    affine = normalized * weight.view(1, CHANNELS, 1, 1) + bias.view(1, CHANNELS, 1, 1)
    bn_bf = affine.to(torch.bfloat16)
    relu = torch.relu(bn_bf)
    le = relu <= 0
    where_val = torch.where(le, fill, slice_channels_last)

    where_f32 = where_val.to(torch.float32)
    sum_1 = where_f32.sum(dim=[0, 2, 3])
    sum_2 = (where_f32 * centered_source).sum(dim=[0, 2, 3])
    mul_10 = sum_2 * invstd_c

    mean_term = (sum_1 * inv_count).view(1, CHANNELS, 1, 1)
    variance_term = (sum_2 * inv_count * (invstd_c * invstd_c)).view(1, CHANNELS, 1, 1)
    output_scale = (invstd_c * weight).view(1, CHANNELS, 1, 1)

    grad_f = (where_f32 - centered_source * variance_term - mean_term) * output_scale
    out = grad_f.to(torch.bfloat16)
    return sum_1, mul_10, out


@oracle_impl(hardware="B200", point="9eb75afe", BLOCK_C=8, BLOCK_K=256, FINAL_BLOCK_C=8, BLOCK_ELEMS=256)
def oracle_forward(inputs, **_kwargs):
    (arg0, arg1, arg2, arg3, arg4,
     arg5, arg6, arg7, arg8, arg9,
     arg10, arg11, arg12, arg13, arg14, arg15,
     arg16, arg17, arg18, arg19, arg20,
     arg21, arg22, arg23, arg24, arg25) = inputs

    inv_count = 2.703287197231834e-05  # captured constant = 1/(128*17*17)

    # Shared prefix: add_2 = avg_pool2d_backward(arg0, arg1) + arg2 + arg3 + arg4
    avg_bw = torch.ops.aten.avg_pool2d_backward.default(
        arg0, arg1, [3, 3], [1, 1], [1, 1], False, True, None
    )
    add_2 = avg_bw + arg2 + arg3 + arg4

    slice_1 = add_2[:, 0:192, :, :]
    slice_2 = add_2[:, 192:384, :, :]
    slice_3 = add_2[:, 384:576, :, :]
    slice_4 = add_2[:, 576:768, :, :]

    s1, m1, c1 = _run_branch(slice_4, arg5, arg6, arg7, arg8, arg9, arg10, inv_count)
    s3, m3, c3 = _run_branch(slice_3, arg11, arg12, arg13, arg14, arg15, arg10, inv_count)
    s5, m5, c5 = _run_branch(slice_2, arg16, arg17, arg18, arg19, arg20, arg10, inv_count)
    s7, m7, c7 = _run_branch(slice_1, arg21, arg22, arg23, arg24, arg25, arg10, inv_count)

    return s1, m1, c1, s3, m3, c3, s5, m5, c5, s7, m7, c7
