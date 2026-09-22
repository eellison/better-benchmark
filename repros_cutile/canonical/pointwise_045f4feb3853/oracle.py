"""cuTile port of pointwise_045f4feb3853: bf16 GPTNeo head transpose clone.

Ports the Triton `_head_transpose_clone_kernel` to cuTile. Source is a bf16
[A*B, C, D] view of [A,B,C,D]; the output view is a contiguous [A,C,B,D]
clone whose base storage this kernel writes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_transpose_clone_kernel(
    src,  # (A, B, C, D)
    dst,  # (A, C, B, D)
    BLOCK_D: ct.Constant[int],
):
    a = ct.bid(0)
    b = ct.bid(1)
    c = ct.bid(2)
    # Load (1, 1, 1, BLOCK_D) from src and store to dst.
    values = ct.load(src, index=(a, b, c, 0), shape=(1, 1, 1, BLOCK_D))
    # Reshape from (1,1,1,BLOCK_D) to (1,1,1,BLOCK_D) - the dst has axes
    # arranged as (A, C, B, D) so store at (a, c, b, 0).
    ct.store(dst, index=(a, c, b, 0), tile=values)


@oracle_impl(hardware="B200", point="e6f344ac", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    A = int(_shape_param_0[0])
    B = int(_shape_param_0[1])
    C = int(_shape_param_0[2])
    D = int(_shape_param_0[3])

    clone = torch.empty_strided(
        (A, C, B, D),
        (C * B * D, B * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # View source as [A, B, C, D] to match the layout used by the original.
    src4 = arg0_1.view(A, B, C, D)
    stream = torch.cuda.current_stream()
    # BLOCK_D = D so we get one tile per (a, b, c).
    ct.launch(
        stream,
        (A, B, C),
        _head_transpose_clone_kernel,
        (src4, clone, D),
    )

    view_2 = clone.view(tuple(_shape_param_2))
    return view_2, view_2.permute(1, 0)
