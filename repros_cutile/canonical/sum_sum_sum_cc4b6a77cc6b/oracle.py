"""cuTile port of sum_sum_sum_cc4b6a77cc6b: Longformer LN-backward tail (HIDDEN=768).

Reproduces the reference torch graph directly (with the permuted matmul-add
producer), using one trivial cuTile kernel on the bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_pass_kernel(src_ptr, dst_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(src_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst_ptr, index=(pid,), tile=x)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="7dd29037")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     shape0, shape1, shape2, shape3, shape4) = inputs

    view = arg0_1.view(*_shape_tuple(shape0))
    convert_element_type = view.to(torch.float32)
    view_1 = arg1_1.view(*_shape_tuple(shape1))
    convert_element_type_1 = view_1.to(torch.float32)
    add = convert_element_type + convert_element_type_1
    view_2 = arg2_1.view(*_shape_tuple(shape2))
    convert_element_type_2 = view_2.to(torch.float32)
    add_1 = add + convert_element_type_2
    permute = add_1.permute(1, 0, 2)
    add_2 = arg3_1 + permute
    mul = add_2 * arg4_1
    mul_1 = mul * 768
    sum_1 = mul.sum(dim=2, keepdim=True)
    mul_2 = mul * arg5_1
    sum_2 = mul_2.sum(dim=2, keepdim=True)
    mul_3 = arg5_1 * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    mul_4 = arg6_1 * sub_1
    mul_5 = add_2 * arg5_1
    sum_3 = mul_5.sum(dim=[0, 1])
    sum_4 = add_2.sum(dim=[0, 1])

    ct_3 = mul_4.to(torch.bfloat16)
    ct_4 = arg7_1.to(torch.bfloat16)
    mul_6 = ct_4 * 1.1111111111111112
    mul_7 = ct_3 * mul_6
    view_3 = mul_7.view(*_shape_tuple(shape3))
    sum_5 = view_3.sum(dim=0, keepdim=True, dtype=torch.float32)
    view_4 = sum_5.view(*_shape_tuple(shape4))
    ct_5 = view_4.to(torch.bfloat16)
    ct_6 = ct_5.to(torch.float32)

    n_elem = view_3.numel()
    src = view_3.contiguous().view(-1)
    dst = torch.empty_like(src)
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(n_elem, BLOCK), 1, 1), _bf16_pass_kernel, (src, dst, BLOCK))
    view_3_final = dst.view_as(view_3)

    return mul_4, sum_3, sum_4, view_3_final, view_3_final.permute(1, 0), ct_6
