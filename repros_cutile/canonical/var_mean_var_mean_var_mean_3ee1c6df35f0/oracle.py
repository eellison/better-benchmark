"""cuTile port of var_mean_var_mean_var_mean_3ee1c6df35f0: RepVGG 3-branch
BN training with running-stat updates.

Torch handles the compute. Trivial pass-through cuTile cast is dropped.
"""

import torch

from oracle_harness import oracle_impl


def _bn_branch(x, running_mean, running_var, weight, bias, scale_var):
    conv = x.to(torch.float32)
    var, mean = torch.var_mean(conv, dim=[0, 2, 3], correction=0, keepdim=True)
    add = var + 1.0e-5
    rsqrt = torch.rsqrt(add)
    sub = x - mean
    normalized = sub * rsqrt
    squeeze_m = mean.view(-1)
    squeeze_r = rsqrt.view(-1)
    squeeze_v = var.view(-1)
    c = x.shape[1]
    w_ = weight.view(c, 1, 1)
    b_ = bias.view(c, 1, 1)
    affine = normalized * w_ + b_
    add_1 = squeeze_m * 0.1 + running_mean * 0.9
    add_2 = (squeeze_v * scale_var * 0.1) + running_var * 0.9
    return affine.to(torch.bfloat16), squeeze_m, squeeze_r, add_1, add_2


@oracle_impl(hardware="B200", point="3009c407", block_e=1024, block_c=8, block=1024, raw_stats=False, fix_d6dd=False)
@oracle_impl(hardware="B200", point="d6dd53ac", block_e=1024, block_c=8, block=1024, raw_stats=True, fix_d6dd=True)
@oracle_impl(hardware="B200", point="9a6632a5", block_e=1024, block_c=8, block=1024, raw_stats=False, fix_d6dd=False)
def oracle_forward(inputs, **_kwargs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
     arg5_1, arg6_1, arg7_1, arg8_1, arg9_1,
     arg10_1, arg11_1, arg12_1, arg13_1, arg14_1) = inputs

    SCALE_VAR = 1.0000398612827361

    branch0_out, squeeze0, squeeze0_r, add_1, add_2 = _bn_branch(
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, SCALE_VAR)
    branch1_out, squeeze1, squeeze1_r, add_5, add_6 = _bn_branch(
        arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, SCALE_VAR)
    branch2_out, squeeze2, squeeze2_r, add_9, add_10 = _bn_branch(
        arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, SCALE_VAR)

    added_12 = branch1_out + branch2_out
    added_13 = added_12 + branch0_out
    relu = torch.relu(added_13)

    c = arg0_1.shape[1]
    unsqueeze_14 = squeeze2.view(1, c, 1, 1)
    unsqueeze_17 = squeeze1.view(1, c, 1, 1)
    unsqueeze_20 = squeeze0.view(1, c, 1, 1)

    copy_ = arg1_1.copy_(add_1)
    copy__1 = arg2_1.copy_(add_2)
    copy__2 = arg6_1.copy_(add_5)
    copy__3 = arg7_1.copy_(add_6)
    copy__4 = arg11_1.copy_(add_9)
    copy__5 = arg12_1.copy_(add_10)

    return (squeeze0_r, squeeze1_r, squeeze2_r, relu,
            unsqueeze_14, unsqueeze_17, unsqueeze_20,
            copy_, copy__1, copy__2, copy__3, copy__4, copy__5)
