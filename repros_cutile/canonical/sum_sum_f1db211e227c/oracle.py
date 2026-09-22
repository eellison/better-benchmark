"""cuTile port of sum_sum_f1db211e227c: GhostNet gated spatial reduction.

Per-(batch, channel) reduce arg0 * arg1 over (h,w). Then hardswish-like gate:
where(-3 < arg2 < 3, sum*1/6, arg3). Sum reduction over batch+spatial for
the final f32[C] output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _spatial_reduce_kernel(
    a_ptr,       # bf16 [N*C, H*W]
    b_ptr,       # bf16 [N*C, H*W]
    out_ptr,     # f32  [N*C]
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)
    a = ct.load(a_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    b = ct.load(b_ptr, index=(row, 0), shape=(1, BLOCK_HW))
    prod_f = ct.astype(a, ct.float32) * ct.astype(b, ct.float32)
    # Triton does (lhs*rhs).to(bf16).to(f32) element-wise, then sum
    prod_rounded = ct.astype(ct.astype(prod_f, ct.bfloat16), ct.float32)
    total = ct.sum(prod_rounded)
    # Also round the final sum to bf16 and back to f32
    total_rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(row,), tile=ct.reshape(total_rounded, (1,)))


def _run(inputs, *, BLOCK_HW):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n, c, h, w = int(arg0_1.shape[0]), int(arg0_1.shape[1]), int(arg0_1.shape[2]), int(arg0_1.shape[3])
    device = arg0_1.device
    rows = n * c
    hw = h * w

    # Get 2D layout. Input has channels-last stride, but we want per-(N,C) view.
    a_2d = arg0_1.contiguous().view(rows, hw)
    b_2d = arg1_1.contiguous().view(rows, hw)

    sum_1 = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (rows, 1, 1), _spatial_reduce_kernel,
              (a_2d, b_2d, sum_1, BLOCK_HW))

    sum_1_view = sum_1.view(n, c, 1, 1)
    convert_element_type_1 = sum_1_view.to(torch.bfloat16)
    convert_element_type_2 = convert_element_type_1.to(torch.float32)

    convert_element_type_arg2 = arg2_1.to(torch.float32)  # [512, 120, 1, 1]
    gt = convert_element_type_arg2 > -3.0
    lt = convert_element_type_arg2 < 3.0
    bitwise_and = gt & lt
    mul_1 = convert_element_type_2 * 0.16666666666666666
    where = torch.where(bitwise_and, mul_1, arg3_1)
    convert_element_type_3 = where.to(torch.bfloat16)
    sum_2 = convert_element_type_3.sum(dim=[0, 2, 3])  # bf16 [C]
    convert_element_type_4 = sum_2.to(torch.float32)  # f32 [C]

    return convert_element_type_arg2, convert_element_type_3, convert_element_type_4


@oracle_impl(hardware="B200", point="8f106dba", BLOCK_HW=1024)
@oracle_impl(hardware="B200", point="f063e905", BLOCK_HW=256)
@oracle_impl(hardware="B200", point="ae67748a", BLOCK_HW=256)
@oracle_impl(hardware="B200", point="2514b9d8", BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_HW):
    return _run(inputs, BLOCK_HW=BLOCK_HW)
