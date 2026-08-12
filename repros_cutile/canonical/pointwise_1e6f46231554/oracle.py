"""cuTile port of pointwise_1e6f46231554: attention head layout transpose clone.

Reshapes the flat bf16 [B*S, H*D] input into [B, S, H, D], transposes S and D to
produce contiguous [B*H, D, S] and returns both the clone and its S<->D permute alias.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_transpose_clone_kernel(
    src,  # bf16 (B, S, H, D)
    dst,  # bf16 (B, H, D, S)
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s_block = ct.bid(2)

    values = ct.load(src, index=(b, s_block, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    values = ct.reshape(values, (BLOCK_S, BLOCK_D))
    values_t = ct.transpose(values)
    values_t = ct.reshape(values_t, (1, 1, BLOCK_D, BLOCK_S))
    ct.store(dst, index=(b, h, 0, s_block), tile=values_t)


@oracle_impl(hardware="B200", point="631b8e39", BLOCK_S=64, BLOCK_D=64)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_S=64, BLOCK_D=64)
@oracle_impl(hardware="B200", point="cb7fdfdf", BLOCK_S=64, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b63e0b0f", BLOCK_S=64, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    B = int(_shape_param_2[0])
    H = int(_shape_param_2[1])
    D = int(_shape_param_2[2])
    S = int(_shape_param_2[3])

    view_2 = torch.empty_strided(
        tuple(_shape_param_3),
        (D * S, S, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    src4 = arg0_1.view(B, S, H, D)
    dst4 = view_2.view(B, H, D, S)

    stream = torch.cuda.current_stream()
    grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
    ct.launch(
        stream,
        grid,
        _head_layout_transpose_clone_kernel,
        (src4, dst4, BLOCK_S, BLOCK_D),
    )

    return view_2, view_2.permute(0, 2, 1)
