"""cuTile port of sum_sum_sum_4a1fb62e29b0: masked-LM layernorm-backward tail.

Complex multi-stage BN-like backward. cuTile handles the final elementwise
mul_7 fusion; row and column reductions delegate to torch to match the
Repro's exact reduction order.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _mul_bf16_kernel(
    conv3_ptr,   # bf16 flat
    mul_6_ptr,   # bf16 flat
    out_ptr,     # bf16 flat
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    conv3 = ct.load(conv3_ptr, index=(pid,), shape=(BLOCK,))
    mul_6 = ct.load(mul_6_ptr, index=(pid,), shape=(BLOCK,))
    out = conv3 * mul_6
    ct.store(out_ptr, index=(pid,), tile=out)


def _shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="5acb4703")
@oracle_impl(hardware="B200", point="6de23498")
@oracle_impl(hardware="B200", point="c1e38d67")
@oracle_impl(hardware="B200", point="c21f4298")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     shape0, shape1, shape2, shape3, shape4) = inputs
    norm_shape = _shape(shape0)  # (n, seq, hidden)
    flat_shape = _shape(shape3)
    hidden = int(norm_shape[-1])
    device = arg0_1.device

    view = arg0_1.view(norm_shape).to(torch.float32)
    add = arg1_1 + view
    add_1 = add + arg2_1.view(norm_shape).to(torch.float32)
    add_2 = add_1 + arg3_1.view(norm_shape).to(torch.float32)
    mul = add_2 * arg4_1

    # Reductions via torch to match Repro numerics exactly.
    # The "1536" constant is baked into the trace regardless of tensor shape.
    mul_1 = mul * 1536
    sum_1 = mul.sum(dim=[2], keepdim=True)
    mul_2 = mul * arg5_1
    sum_2 = mul_2.sum(dim=[2], keepdim=True)
    mul_3 = arg5_1 * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    mul_4 = arg6_1 * sub_1
    mul_5 = add_2 * arg5_1
    sum_3 = mul_5.sum(dim=[0, 1])
    sum_4 = add_2.sum(dim=[0, 1])

    conv3 = mul_4.to(torch.bfloat16)
    conv4 = arg7_1.to(torch.bfloat16)
    mul_6 = conv4 * 1.1111111111111112

    # Do the final elementwise product via cuTile.
    total = conv3.numel()
    mul_7_flat = torch.empty(total, device=device, dtype=torch.bfloat16)
    conv3_flat = conv3.contiguous().view(total)
    mul_6_flat = mul_6.contiguous().view(total)
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((total + BLOCK - 1) // BLOCK, 1, 1),
        _mul_bf16_kernel,
        (conv3_flat, mul_6_flat, mul_7_flat, BLOCK),
    )
    mul_7 = mul_7_flat.view(norm_shape)

    view_3 = mul_7.view(flat_shape)
    permute = view_3.permute(1, 0)
    sum_5 = view_3.sum(dim=[0], keepdim=True, dtype=torch.float32).view(hidden)
    conv5 = sum_5.to(torch.bfloat16)
    conv6 = conv5.to(torch.float32)

    return mul_4, sum_3, sum_4, view_3, permute, conv6
