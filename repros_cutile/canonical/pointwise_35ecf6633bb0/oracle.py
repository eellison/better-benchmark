"""cuTile port of pointwise_35ecf6633bb0: attention head layout clone + permute alias.

Reshapes the flat bf16 input into [B, S, H, D], writes it as contiguous [B, H, S, D],
and returns the [B*H, D, S] permuted alias.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_clone_kernel(
    src,  # bf16 (B, S, H, D)
    dst,  # bf16 (B, H, S, D)
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s_block = ct.bid(2)

    values = ct.load(src, index=(b, s_block, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    values = ct.reshape(values, (1, 1, BLOCK_S, BLOCK_D))
    ct.store(dst, index=(b, h, s_block, 0), tile=values)


@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_S=32, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    B = int(_shape_param_0[0])
    S = int(_shape_param_0[1])
    D = int(_shape_param_2[2])
    H = int(arg0_1.shape[1]) // D

    clone = torch.empty_strided(
        (B, H, S, D),
        (H * S * D, S * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    src4 = arg0_1.view(B, S, H, D)

    stream = torch.cuda.current_stream()
    grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
    ct.launch(
        stream,
        grid,
        _head_layout_clone_kernel,
        (src4, clone, BLOCK_S, BLOCK_D),
    )

    return clone.view(tuple(_shape_param_2)).permute(0, 2, 1)
