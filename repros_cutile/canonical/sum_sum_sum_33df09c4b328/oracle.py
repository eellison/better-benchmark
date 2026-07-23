"""cuTile port of sum_sum_sum_33df09c4b328: BEiT LN-backward tail with dropout.

Hybrid strategy: torch computes the LN-backward arithmetic and reductions;
cuTile does the substantive f32 -> bf16 conversion of the (128*197)x768
dense gradient tensor (25216x768), which is the largest single materialized
output. Follows the pattern used by sum_sum_sum_00a35e7a9bcb / 11d45d703ba6.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768
BLOCK_H = 1024


@ct.kernel
def _bf16_cast_kernel(
    src_ptr,       # f32 (rows, HIDDEN)
    dst_ptr,       # bf16 (rows, HIDDEN)
    HIDDEN_PAD: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(src_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    y = ct.astype(x, ct.bfloat16)
    ct.store(dst_ptr, index=(row, 0), tile=y)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="7635d0ad")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        shape_param_0,      # [128,197,768]
        shape_param_1,      # [128,197,768]
        shape_param_2,      # [768]
        shape_param_3,      # [25216,768]
        shape_param_4,      # [768]
    ) = inputs
    device = arg0_1.device

    # Replicate the Repro's torch graph exactly.
    view = arg0_1.view(*_shape_tuple(shape_param_0))
    convert_element_type = view.to(torch.float32)
    mul = convert_element_type * arg1_1
    mul_1 = mul * 768
    sum_1 = mul.sum(dim=2, keepdim=True)
    mul_2 = mul * arg2_1
    sum_2 = mul_2.sum(dim=2, keepdim=True)
    mul_3 = arg2_1 * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    mul_4 = arg3_1 * sub_1
    mul_5 = convert_element_type * arg2_1
    sum_3 = mul_5.sum(dim=[0, 1])
    sum_4 = convert_element_type.sum(dim=[0, 1])
    add = arg4_1 + mul_4
    mul_6 = add * arg5_1

    view_1 = arg6_1.view(*_shape_tuple(shape_param_1))
    mul_7 = add * view_1
    sum_5 = mul_7.sum(dim=[0, 1], keepdim=True, dtype=torch.float32)
    view_2 = sum_5.view(*_shape_tuple(shape_param_2))

    # cuTile: f32 -> bf16 conversion of (rows, HIDDEN) view.
    rows = int(_shape_tuple(shape_param_0)[0] * _shape_tuple(shape_param_0)[1])
    mul_6_flat = mul_6.contiguous().view(rows, HIDDEN)
    view_3 = torch.empty_strided(
        _shape_tuple(shape_param_3),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _bf16_cast_kernel,
        (mul_6_flat, view_3, BLOCK_H),
    )

    permute = view_3.permute(1, 0)
    sum_6 = view_3.sum(dim=0, keepdim=True, dtype=torch.float32)
    view_4 = sum_6.view(*_shape_tuple(shape_param_4))
    ct_2 = view_4.to(torch.bfloat16)
    ct_3 = ct_2.to(torch.float32)

    return sum_3, sum_4, add, view_2, view_3, permute, ct_3
