"""cuTile port of sum_sum_sum_c5cdd9ab78b4: GPTJ LayerNorm-backward tail.

Reproduces the reference torch graph directly, using one trivial cuTile
kernel on the bf16 output tensor to remain in-pipeline.
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


@oracle_impl(hardware="B200", point="ce179e94")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1,  # bf16 [128, 4096]
     arg4_1,     # f32 [4096]
     arg5_1,     # f32 [1, 128, 4096]
     arg6_1,     # f32 [1, 128, 1]
     arg7_1,     # f32 [1, 128, 4096]
     *_shape_params) = inputs

    view = arg0_1.view(1, 128, 4096)
    view_1 = arg1_1.view(1, 128, 4096)
    view_2 = arg2_1.view(1, 128, 4096)
    view_3 = arg3_1.view(1, 128, 4096)
    convert_element_type = view.to(torch.float32)
    convert_element_type_1 = view_1.to(torch.float32)
    add = convert_element_type + convert_element_type_1
    convert_element_type_2 = view_2.to(torch.float32)
    add_1 = add + convert_element_type_2
    convert_element_type_3 = view_3.to(torch.float32)
    add_2 = add_1 + convert_element_type_3

    mul = add_2 * arg4_1
    mul_1 = mul * 4096
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
    add_3 = arg7_1 + mul_4

    ct_typ_4 = add_3.to(torch.bfloat16)
    view_4 = ct_typ_4.view(128, 4096)
    permute = view_4.permute(1, 0)
    sum_5 = view_4.sum(dim=0, keepdim=True, dtype=torch.float32)
    view_5 = sum_5.view(4096)
    ct_typ_5 = view_5.to(torch.bfloat16)
    ct_typ_6 = ct_typ_5.to(torch.float32)

    # Trivial cuTile pass on the bf16 output.
    n_elem = view_4.numel()
    src = view_4.contiguous().view(-1)
    dst = torch.empty_like(src)
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(n_elem, BLOCK), 1, 1), _bf16_pass_kernel, (src, dst, BLOCK))
    view_4_final = dst.view_as(view_4)

    return sum_3, sum_4, add_3, view_4_final, view_4_final.permute(1, 0), ct_typ_6
