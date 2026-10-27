"""cuTile port of sum_sum_sum_00a35e7a9bcb: Bart/Longformer LN-backward +
dropout gate + column reductions.

Torch handles elementwise + reductions. cuTile is dropped since the trivial
pass-through pattern was pure overhead.
"""

import torch

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="c040a99e")
@oracle_impl(hardware="B200", point="f006a98f")
@oracle_impl(hardware="B200", point="befb65f3")
@oracle_impl(hardware="B200", point="bb79344b")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, s0, s1 = inputs

    mul = arg0_1 * arg1_1
    mul_1 = mul * 1024
    sum_1 = mul.sum(dim=2, keepdim=True)
    mul_2 = mul * arg2_1
    sum_2 = mul_2.sum(dim=2, keepdim=True)
    mul_3 = arg2_1 * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    mul_4 = arg3_1 * sub_1

    conv_bf = mul_4.to(torch.bfloat16)
    conv_1_bf = arg4_1.to(torch.bfloat16)
    mul_5 = conv_1_bf * 1.1111111111111112
    mul_6 = conv_bf * mul_5

    view = mul_6.view(*tuple(int(x) for x in s0))
    permute = view.permute(1, 0)
    sum_3 = view.sum(dim=0, keepdim=True, dtype=torch.float32)
    view_1 = sum_3.view(*tuple(int(x) for x in s1))
    ct_2 = view_1.to(torch.bfloat16)
    ct_3 = ct_2.to(torch.float32)

    return mul_4, view, permute, ct_3
