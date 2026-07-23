"""cuTile port of pointwise_733dafce05a6: TrOCR/XGLM scaled attention-head layout.

NEW_PATTERN: view bf16 [B*S, H*D] as [B, S, H, D], scale by 0.125, permute to
[B, H, S, D], and materialize contiguously with shape [B*H, S, D].

We reshape once to expose the four logical axes and use a 3D grid
(bh, s_block, d_block) so cuTile writes contiguous [B*H, S, D] tiles from
the strided [B, S, H, D] input via array-space indexing on the reshaped
input tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scaled_head_layout_kernel(
    src,   # bf16 [B, S, H, D]
    dst,   # bf16 [B, H, S, D]
    B: ct.Constant[int],
    S: ct.Constant[int],
    H: ct.Constant[int],
    D: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    bh = ct.bid(0)
    s_block = ct.bid(1)
    b = bh // H
    h = bh - b * H
    # Load a (1, BLOCK_S, 1, BLOCK_D) tile from src at (b, s_block, h, 0)
    values = ct.load(src, index=(b, s_block, h, 0), shape=(1, BLOCK_S, 1, BLOCK_D))
    values_f = ct.astype(values, ct.float32)
    scaled = ct.astype(values_f * 0.125, ct.bfloat16)
    # Store into dst at (b, h, s_block, 0) with shape (1, 1, BLOCK_S, BLOCK_D)
    scaled_4d = ct.reshape(scaled, (1, 1, BLOCK_S, BLOCK_D))
    ct.store(dst, index=(b, h, s_block, 0), tile=scaled_4d)


@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_S=16, BLOCK_D=64)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_S=32, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    B = int(_shape_param_0[0])
    S = int(_shape_param_0[1])
    H = int(_shape_param_1[2])
    D = int(_shape_param_1[3])

    src4 = arg0_1.view(B, S, H, D)
    out = torch.empty(
        tuple(int(dim) for dim in _shape_param_2),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    dst4 = out.view(B, H, S, D)

    grid = (B * H, (S + BLOCK_S - 1) // BLOCK_S, 1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _scaled_head_layout_kernel,
        (src4, dst4, B, S, H, D, BLOCK_S, BLOCK_D),
    )
    return out
