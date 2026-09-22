"""cuTile port of sum_sum_sum_2b1b93ca6e56: functorch_dp_cifar10 GroupNorm backward.

Complex dual-GN-backward composition. Uses torch for most ops (which are
graph-capturable) and a cuTile kernel for the mask + relu-backward gate
(where(le(relu, 0), zero_scalar, add)).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_gate_kernel(
    add_ptr,           # f32 [N]
    relu_input_ptr,    # f32 [N] — this is add_3 that goes into relu
    fill_ptr,          # f32 [1] scalar
    out_ptr,           # f32 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    add = ct.load(add_ptr, index=(pid,), shape=(BLOCK,))
    rel = ct.load(relu_input_ptr, index=(pid,), shape=(BLOCK,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    # le(relu(rel), 0) — relu(x) <= 0 iff x <= 0
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    le = rel <= zero
    fill_broad = ct.zeros((BLOCK,), dtype=ct.float32) + ct.reshape(fill, (1,))
    result = ct.where(le, fill_broad, add)
    ct.store(out_ptr, index=(pid,), tile=result)


def _shape(shape):
    return tuple(int(d) for d in shape)


def _run(inputs, **kwargs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, *shapes) = inputs
    device = arg0_1.device
    shape_params = shapes  # 21 shape params

    convert_element_type = arg0_1.to(torch.float32)
    add = arg1_1 + convert_element_type
    convert_element_type_1 = arg2_1.to(torch.float32)
    view = convert_element_type_1.view(_shape(shape_params[0]))
    sub = view - arg3_1
    mul = sub * arg4_1
    view_1 = mul.view(_shape(shape_params[1]))
    unsqueeze_2 = arg5_1.view(1, -1, 1, 1)
    mul_1 = view_1 * unsqueeze_2
    unsqueeze_5 = arg6_1.view(1, -1, 1, 1)
    add_1 = mul_1 + unsqueeze_5

    convert_element_type_2 = arg7_1.to(torch.float32)
    view_2 = convert_element_type_2.view(_shape(shape_params[2]))
    sub_1 = view_2 - arg8_1
    mul_2 = sub_1 * arg9_1
    view_3 = mul_2.view(_shape(shape_params[3]))
    unsqueeze_8 = arg10_1.view(1, -1, 1, 1)
    mul_3 = view_3 * unsqueeze_8
    unsqueeze_11 = arg11_1.view(1, -1, 1, 1)
    add_2 = mul_3 + unsqueeze_11
    add_3 = add_1 + add_2

    # Use cuTile for the where(le(relu, 0), fill, add) op
    numel = int(add.numel())
    add_flat = add.contiguous().view(numel)
    add_3_flat = add_3.contiguous().view(numel)
    fill_1d = arg12_1.view(1).contiguous()
    where_out = torch.empty(numel, device=device, dtype=torch.float32)
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _relu_gate_kernel,
              (add_flat, add_3_flat, fill_1d, where_out, BLOCK))
    where = where_out.view(add.shape)

    # Continue with the rest of the computation in torch
    mul_4 = where * convert_element_type_2
    view_4 = mul_4.view(_shape(shape_params[4]))
    sum_1 = view_4.sum(dim=[2])
    view_5 = where.view(_shape(shape_params[5]))
    sum_2 = view_5.sum(dim=[2])

    unsqueeze = arg5_1.view(1, -1)
    unsqueeze_6 = arg10_1.view(1, -1)

    mul_5 = sum_1 * unsqueeze_6
    view_6 = mul_5.view(_shape(shape_params[6]))
    sum_3 = view_6.sum(dim=[2])
    mul_6 = sum_2 * unsqueeze_6
    view_7 = mul_6.view(_shape(shape_params[7]))
    sum_4 = view_7.sum(dim=[2])

    squeeze = arg9_1.squeeze(-1).squeeze(-1)  # [128, 32]
    unsqueeze_12 = squeeze.unsqueeze(-1)
    view_8 = arg10_1.view(_shape(shape_params[8]))
    mul_7 = unsqueeze_12 * view_8
    squeeze_1 = arg8_1.squeeze(-1).squeeze(-1)
    mul_8 = sum_4 * squeeze_1
    sub_2 = mul_8 - sum_3
    mul_9 = sub_2 * squeeze
    mul_10 = mul_9 * squeeze
    mul_11 = mul_10 * squeeze
    mul_12 = mul_11 * 0.015625
    neg = -mul_12
    mul_13 = neg * squeeze_1
    mul_14 = sum_4 * squeeze
    mul_15 = mul_14 * 0.015625
    sub_3 = mul_13 - mul_15
    unsqueeze_13 = mul_7.unsqueeze(-1)
    unsqueeze_15 = mul_12.unsqueeze(-1).unsqueeze(-1)
    unsqueeze_17 = sub_3.unsqueeze(-1).unsqueeze(-1)

    view_9 = where.view(_shape(shape_params[9]))
    mul_16 = view_9 * unsqueeze_13
    mul_17 = view_2 * unsqueeze_15
    add_4 = mul_16 + mul_17
    add_5 = add_4 + unsqueeze_17
    view_10 = add_5.view(_shape(shape_params[10]))

    view_11 = sum_1.view(_shape(shape_params[11]))
    view_12 = sum_2.view(_shape(shape_params[12]))
    unsqueeze_18 = squeeze_1.unsqueeze(-1)
    mul_18 = view_12 * unsqueeze_18
    sub_4 = view_11 - mul_18
    mul_19 = sub_4 * unsqueeze_12
    sum_5 = mul_19.sum(dim=[0])
    view_13 = sum_5.view(_shape(shape_params[13]))
    sum_6 = sum_2.sum(dim=[0])
    convert_element_type_3 = view_10.to(torch.bfloat16)

    # Second half — for the arg1_1 group (parallel)
    mul_20 = where * convert_element_type_1
    view_14 = mul_20.view(_shape(shape_params[14]))
    sum_7 = view_14.sum(dim=[2])
    mul_21 = sum_7 * unsqueeze
    view_15 = mul_21.view(_shape(shape_params[15]))
    sum_8 = view_15.sum(dim=[2])
    mul_22 = sum_2 * unsqueeze
    view_16 = mul_22.view(_shape(shape_params[16]))
    sum_9 = view_16.sum(dim=[2])

    squeeze_2 = arg4_1.squeeze(-1).squeeze(-1)
    unsqueeze_19 = squeeze_2.unsqueeze(-1)
    view_17 = arg5_1.view(_shape(shape_params[17]))
    mul_23 = unsqueeze_19 * view_17
    squeeze_3 = arg3_1.squeeze(-1).squeeze(-1)
    mul_24 = sum_9 * squeeze_3
    sub_5 = mul_24 - sum_8
    mul_25 = sub_5 * squeeze_2
    mul_26 = mul_25 * squeeze_2
    mul_27 = mul_26 * squeeze_2
    mul_28 = mul_27 * 0.015625
    neg_1 = -mul_28
    mul_29 = neg_1 * squeeze_3
    mul_30 = sum_9 * squeeze_2
    mul_31 = mul_30 * 0.015625
    sub_6 = mul_29 - mul_31
    unsqueeze_20 = mul_23.unsqueeze(-1)
    unsqueeze_22 = mul_28.unsqueeze(-1).unsqueeze(-1)
    unsqueeze_24 = sub_6.unsqueeze(-1).unsqueeze(-1)
    mul_32 = view_9 * unsqueeze_20
    mul_33 = view * unsqueeze_22
    add_6 = mul_32 + mul_33
    add_7 = add_6 + unsqueeze_24
    view_18 = add_7.view(_shape(shape_params[18]))
    view_19 = sum_7.view(_shape(shape_params[19]))
    unsqueeze_25 = squeeze_3.unsqueeze(-1)
    mul_34 = view_12 * unsqueeze_25
    sub_7 = view_19 - mul_34
    mul_35 = sub_7 * unsqueeze_19
    sum_10 = mul_35.sum(dim=[0])
    view_20 = sum_10.view(_shape(shape_params[20]))
    sum_11 = sum_2.sum(dim=[0])
    convert_element_type_4 = view_18.to(torch.bfloat16)

    return view_13, sum_6, convert_element_type_3, view_20, sum_11, convert_element_type_4


@oracle_impl(hardware="B200", point="9705fec3")
@oracle_impl(hardware="B200", point="95a30d45")
@oracle_impl(hardware="B200", point="49b9f54f")
def oracle_forward(inputs):
    return _run(inputs)
