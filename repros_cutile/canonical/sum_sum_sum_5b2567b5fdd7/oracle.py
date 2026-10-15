"""cuTile port of sum_sum_sum_5b2567b5fdd7: LN-backward with column
reductions + bf16 midpoint.

Torch handles the elementwise LN-backward math and reductions (all
straight ops); cuTile handles the final bf16 cast of the residual+update
tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


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


@oracle_impl(hardware="B200", point="0243aeaa",
             ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=256, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="8348e069",
             ROWS_PER_GROUP=16, BLOCK_R=2, BLOCK_C=1024, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="670b1ed7",
             ROWS_PER_GROUP=32, BLOCK_R=2, BLOCK_C=1024, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="18835b6c",
             ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=2048, FINAL_BLOCK_C=2)
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device
    hidden = int(arg2_1.shape[-1])

    view = arg0_1.view(*tuple(int(s) for s in shape0))
    ct_view = view.to(torch.float32)
    mul = ct_view * arg1_1
    # Eager repro hardcodes 192 as the multiplier — same for all shape points.
    mul_1 = mul * 192
    sum_1 = mul.sum(dim=2, keepdim=True)
    mul_2 = mul * arg2_1
    sum_2 = mul_2.sum(dim=2, keepdim=True)
    mul_3 = arg2_1 * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    mul_4 = arg3_1 * sub_1
    mul_5 = ct_view * arg2_1
    sum_3 = mul_5.sum(dim=[0, 1])
    sum_4 = ct_view.sum(dim=[0, 1])
    add = arg4_1 + mul_4

    # bf16 cast via cuTile
    rows = view.numel() // hidden
    hidden_pad = 1 << (hidden - 1).bit_length()
    add_2d = add.contiguous().view(rows, hidden)
    ct_out_2d = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bf16_cast_kernel,
        (add_2d, ct_out_2d, hidden_pad),
    )
    ct_1 = ct_out_2d.view(*tuple(int(s) for s in shape0))
    view_1 = ct_out_2d.view(*tuple(int(s) for s in shape1))
    permute = view_1.permute(1, 0)
    sum_5 = view_1.sum(dim=0, keepdim=True, dtype=torch.float32)
    view_2 = sum_5.view(*tuple(int(s) for s in shape2))
    ct_2 = view_2.to(torch.bfloat16)
    ct_3 = ct_2.to(torch.float32)

    return sum_3, sum_4, add, view_1, permute, ct_3
