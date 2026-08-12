"""cuTile port of pointwise_9b533c851bf2: attention head layout clone.

Reshape input [A*B, C, D] as [A, B, C, D], transpose to [A, C, B, D] output.
Returns view aliases of the clone. All dimensions are power-of-2 here.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_transpose_clone_kernel(
    in_ptr,   # bf16 [A, B, C, D]
    out_ptr,  # bf16 [A, C, B, D]
    BLOCK_C: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    a = ct.bid(0)
    b = ct.bid(1)
    c_tile = ct.bid(2)

    values = ct.load(in_ptr, index=(a, b, c_tile, 0), shape=(1, 1, BLOCK_C, BLOCK_D))
    ct.store(out_ptr, index=(a, c_tile, b, 0), tile=ct.permute(values, (0, 2, 1, 3)))


@oracle_impl(hardware="B200", point="c6cb1dd8", BLOCK_C=8, BLOCK_D=256)
@oracle_impl(hardware="B200", point="226fbbfa", BLOCK_C=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="e6f344ac", BLOCK_C=8, BLOCK_D=128)
@oracle_impl(hardware="B200", point="fb089404", BLOCK_C=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="6c3c2efc", BLOCK_C=16, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    A = int(_shape_param_0[0])
    B = int(_shape_param_0[1])
    C = int(_shape_param_0[2])
    D = int(_shape_param_0[3])

    clone = torch.empty_strided(
        (A, C, B, D), (C * B * D, B * D, D, 1), device=arg0_1.device, dtype=arg0_1.dtype
    )
    src4 = arg0_1.view(A, B, C, D)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (A, B, (C + BLOCK_C - 1) // BLOCK_C),
        _head_transpose_clone_kernel,
        (src4, clone, BLOCK_C, BLOCK_D),
    )

    view_2 = clone.view(tuple(_shape_param_2))
    return view_2, view_2.permute(1, 0)
