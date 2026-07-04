"""cuTile port of sum_sum_sum_81f672a568a0: functorch_dp_cifar10 GN backward with max-pool scatter."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _relu_gate_kernel(
    add_ptr,
    relu_input_ptr,
    fill_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    add = ct.load(add_ptr, index=(pid,), shape=(BLOCK,))
    rel = ct.load(relu_input_ptr, index=(pid,), shape=(BLOCK,))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    le = rel <= zero
    fill_broad = ct.zeros((BLOCK,), dtype=ct.float32) + ct.reshape(fill, (1,))
    result = ct.where(le, fill_broad, add)
    ct.store(out_ptr, index=(pid,), tile=result)


def _shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="05c56638")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     *shapes) = inputs
    device = arg0_1.device
    shape_params = shapes

    convert_element_type = arg0_1.to(torch.float32)
    add = arg1_1 + convert_element_type
    full = torch.zeros(_shape(shape_params[0]), device=device, dtype=torch.float32)
    clone = add.contiguous()
    _unsafe_view = clone.view(_shape(shape_params[1]))
    _low_mem = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg2_1, _shape(shape_params[2]), _shape(shape_params[3]),
        _shape(shape_params[4]), [1, 1], [1, 1]
    )
    clone_1 = _low_mem.contiguous()
    _unsafe_view_1 = clone_1.view(_shape(shape_params[5]))
    scatter_add = full.scatter_add(1, _unsafe_view_1, _unsafe_view)
    view = scatter_add.view(_shape(shape_params[6]))

    convert_element_type_1 = arg3_1.to(torch.float32)
    view_1 = convert_element_type_1.view(_shape(shape_params[7]))
    sub = view_1 - arg4_1
    mul = sub * arg5_1
    view_2 = mul.view(_shape(shape_params[8]))
    unsqueeze = arg6_1.view(1, -1)
    unsqueeze_2 = arg6_1.view(1, -1, 1, 1)
    mul_1 = view_2 * unsqueeze_2
    unsqueeze_5 = arg7_1.view(1, -1, 1, 1)
    add_1 = mul_1 + unsqueeze_5

    # cuTile relu gate: where(le(relu(add_1), 0), arg8_1, view)
    numel = int(view.numel())
    view_flat = view.contiguous().view(numel)
    add_1_flat = add_1.contiguous().view(numel)
    fill_1d = arg8_1.view(1).contiguous()
    where_out = torch.empty(numel, device=device, dtype=torch.float32)
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _relu_gate_kernel,
              (view_flat, add_1_flat, fill_1d, where_out, BLOCK))
    where = where_out.view(view.shape)

    mul_2 = where * convert_element_type_1
    view_3 = mul_2.view(_shape(shape_params[9]))
    sum_1 = view_3.sum(dim=[2])
    view_4 = where.view(_shape(shape_params[10]))
    sum_2 = view_4.sum(dim=[2])

    mul_3 = sum_1 * unsqueeze
    view_5 = mul_3.view(_shape(shape_params[11]))
    sum_3 = view_5.sum(dim=[2])
    mul_4 = sum_2 * unsqueeze
    view_6 = mul_4.view(_shape(shape_params[12]))
    sum_4 = view_6.sum(dim=[2])

    squeeze = arg5_1.squeeze(-1).squeeze(-1)
    unsqueeze_6 = squeeze.unsqueeze(-1)
    view_7 = arg6_1.view(_shape(shape_params[13]))
    mul_5 = unsqueeze_6 * view_7
    squeeze_1 = arg4_1.squeeze(-1).squeeze(-1)
    mul_6 = sum_4 * squeeze_1
    sub_1 = mul_6 - sum_3
    mul_7 = sub_1 * squeeze
    mul_8 = mul_7 * squeeze
    mul_9 = mul_8 * squeeze
    mul_10 = mul_9 * 0.001953125
    neg = -mul_10
    mul_11 = neg * squeeze_1
    mul_12 = sum_4 * squeeze
    mul_13 = mul_12 * 0.001953125
    sub_2 = mul_11 - mul_13
    unsqueeze_7 = mul_5.unsqueeze(-1)
    unsqueeze_9 = mul_10.unsqueeze(-1).unsqueeze(-1)
    unsqueeze_11 = sub_2.unsqueeze(-1).unsqueeze(-1)

    view_8 = where.view(_shape(shape_params[14]))
    mul_14 = view_8 * unsqueeze_7
    mul_15 = view_1 * unsqueeze_9
    add_2 = mul_14 + mul_15
    add_3 = add_2 + unsqueeze_11
    view_9 = add_3.view(_shape(shape_params[15]))

    view_10 = sum_1.view(_shape(shape_params[16]))
    view_11 = sum_2.view(_shape(shape_params[17]))
    unsqueeze_12 = squeeze_1.unsqueeze(-1)
    mul_16 = view_11 * unsqueeze_12
    sub_3 = view_10 - mul_16
    mul_17 = sub_3 * unsqueeze_6
    sum_5 = mul_17.sum(dim=[0])
    view_12 = sum_5.view(_shape(shape_params[18]))
    sum_6 = sum_2.sum(dim=[0])
    convert_element_type_2 = view_9.to(torch.bfloat16)

    return view_12, sum_6, convert_element_type_2
