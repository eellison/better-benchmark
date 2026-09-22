"""cuTile port of sum_sum_sum_cb474de4ede0 (SCATTER_REDUCE): MobileBERT
bf16/fp32 backward with multi-output reductions and scatter-add accumulators.

Uses torch for the scatter_add + reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="4e4a9284")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10) = inputs

    def _as_shape(s): return tuple(int(dim) for dim in s)

    view = arg0_1.view(_as_shape(s0))
    add = arg1_1 + view.to(torch.float32)
    view_1 = arg2_1.view(_as_shape(s1))
    add_1 = add + view_1.to(torch.float32)
    view_2 = arg3_1.view(_as_shape(s2))
    add_2 = add_1 + view_2.to(torch.float32)
    sum_1 = add_2.sum([0, 1], keepdim=True, dtype=torch.float32)
    view_3 = sum_1.view(_as_shape(s3))
    view_4 = arg4_1.view(_as_shape(s4))
    add_3 = view_4 + arg5_1
    add_4 = add_3 + arg6_1
    mul = add_2 * add_4
    mul_1 = add_2 * arg7_1
    sum_2 = mul.sum([0, 1], keepdim=True, dtype=torch.float32)
    view_5 = sum_2.view(_as_shape(s5))
    conv3 = mul_1.to(torch.bfloat16)
    sum_3 = mul_1.sum([0], keepdim=True, dtype=torch.float32)

    full = torch.full(_as_shape(s6), True, dtype=torch.bool, device=arg0_1.device)
    full_1 = torch.zeros(_as_shape(s7), device=arg0_1.device, dtype=torch.float32)
    scatter_1 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_1, full, [arg8_1], mul_1)

    slice_1 = arg9_1[:, 0:128]
    ge = slice_1 >= 0
    lt = slice_1 < 512
    and_ = ge & lt
    ne = slice_1 != -1
    and1 = and_ & ne
    unsq = and1.unsqueeze(-1)
    full_2 = torch.zeros(_as_shape(s8), device=arg0_1.device, dtype=torch.float32)
    scatter_2 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_2, unsq, [slice_1], sum_3)

    view_6 = conv3.view(_as_shape(s9))
    permute = view_6.permute(1, 0)
    sum_4 = view_6.sum([0], keepdim=True, dtype=torch.float32)
    view_7 = sum_4.view(_as_shape(s10))
    conv_4 = view_7.to(torch.bfloat16)
    conv_5 = conv_4.to(torch.float32)

    return view_3, view_5, scatter_1, scatter_2, view_6, permute, conv_5
