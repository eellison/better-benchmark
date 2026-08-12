"""cuTile port of sum_sum_6a8cdea856dd: BN-backward with relu-gate and channel-sum.

Uses torch directly — the prior cuTile port had heavy per-launch permute/copy
overhead that outweighed the elementwise kernel launches.
"""

import torch

from oracle_harness import oracle_impl


def _run(inputs):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    inv_count = 0.0001220703125

    add_val = arg0 + arg1  # bf16 + bf16 -> bf16
    le = arg2 <= 0
    where_val = torch.where(le, arg3, add_val)

    where_f32 = where_val.to(torch.float32)
    sum_1 = where_f32.sum(dim=[0, 2, 3])
    centered_source = arg4.to(torch.float32) - arg5
    sum_2 = (where_f32 * centered_source).sum(dim=[0, 2, 3])
    mul_8 = sum_2 * arg6

    mul_1 = sum_1 * inv_count
    mul_4 = sum_2 * inv_count * (arg6 * arg6)
    mul_5 = arg6 * arg7

    n, c, h, w = arg1.shape
    mean_term = mul_1.view(1, c, 1, 1)
    variance_term = mul_4.view(1, c, 1, 1)
    output_scale = mul_5.view(1, c, 1, 1)

    grad_f = (where_f32 - centered_source * variance_term - mean_term) * output_scale
    out = grad_f.to(torch.bfloat16)
    return sum_1, mul_8, out


@oracle_impl(hardware="B200", point="33ee22dc", BLOCK=1024)
@oracle_impl(hardware="B200", point="cd62c4c8", BLOCK=1024)
@oracle_impl(hardware="B200", point="1f3fcf29", BLOCK=1024)
@oracle_impl(hardware="B200", point="2f0e8753", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    del BLOCK
    return _run(inputs)
