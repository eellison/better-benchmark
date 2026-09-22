"""cuTile port of pointwise_0aaf589f7fad: BERT/T5/MT5 attention-head transpose clone.

Takes the packed input [B*S, H*D] and produces the transposed head-layout clone
[B, H, D, S] via a fused view/permute/expand/clone chain. Returns the [B*H, D, S] view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_transpose_clone_kernel(
    src,      # (B, S, H, D) bf16
    dst,      # (B, H, D, S) bf16
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    d_tile = ct.bid(2)

    # Load (1, BLOCK_S, 1, BLOCK_D) from src at (b, 0, h, d_tile) — BLOCK_S is full S.
    values = ct.load(src, index=(b, 0, h, d_tile), shape=(1, BLOCK_S, 1, BLOCK_D))
    # Reshape to (BLOCK_S, BLOCK_D), transpose to (BLOCK_D, BLOCK_S)
    values2d = ct.reshape(values, (BLOCK_S, BLOCK_D))
    values_t = ct.transpose(values2d)
    values_t_4d = ct.reshape(values_t, (1, 1, BLOCK_D, BLOCK_S))
    ct.store(dst, index=(b, h, d_tile, 0), tile=values_t_4d)


@oracle_impl(hardware="B200", point="631b8e39", BLOCK_S=128, BLOCK_D=64)
@oracle_impl(hardware="B200", point="cb7fdfdf", BLOCK_S=128, BLOCK_D=64)
@oracle_impl(hardware="B200", point="b63e0b0f", BLOCK_S=1024, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    B = int(_shape_param_2[0])
    H = int(_shape_param_2[1])
    D = int(_shape_param_2[2])
    S = int(_shape_param_2[3])

    # arg0_1 is [B*S, H*D]. View as [B, S, H, D]
    src4 = arg0_1.view(B, S, H, D)

    view_2 = torch.empty_strided(
        tuple(_shape_param_3),
        (D * S, S, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    dst4 = view_2.view(B, H, D, S)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, H, ct.cdiv(D, BLOCK_D)),
        _head_layout_transpose_clone_kernel,
        (src4, dst4, BLOCK_S, BLOCK_D),
    )
    return view_2
