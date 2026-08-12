"""cuTile port of pointwise_340cdea6ba6a (NEW_PATTERN): GPT-Neo head layout clone.

Input bf16 [512, 128, 128] is viewed as [A=32, B=16, C=128, D=128] and
transposed to [A, D, B, C] (with A*D=4096, B*C=2048), i.e. permute [0,3,1,2].
Returns (out, out.T).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


A = 32
B = 16
C = 128
D = 128


@ct.kernel
def _head_layout_clone_kernel(
    src,   # bf16 [A, B, C, D]
    dst,   # bf16 [A, D, B, C]
    BLOCK_C: ct.Constant[int],
):
    a = ct.bid(0)
    d = ct.bid(1)
    c_tile = ct.bid(2)

    # Load tile [1, B, BLOCK_C, 1] from src at (a, 0, c_tile*BLOCK_C, d).
    # cuTile tile-space index for (a, b_tile, c_tile, d_tile) with tile
    # (1, B, BLOCK_C, 1) is (a, 0, c_tile, d).
    values = ct.load(src, index=(a, 0, c_tile, d), shape=(1, B, BLOCK_C, 1))
    # Squeeze/permute to (B, BLOCK_C) which we want at dst[a, d, b, c_tile]
    # with shape (1, 1, B, BLOCK_C).
    values_out = ct.reshape(values, (1, 1, B, BLOCK_C))
    ct.store(dst, index=(a, d, 0, c_tile), tile=values_out)


@oracle_impl(hardware="B200", point="e6f344ac", BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_C: int):
    arg0_1, _shape_param_0, _shape_param_1, shape_param_2 = inputs
    src = arg0_1.view(A, B, C, D)
    out = torch.empty_strided(
        (int(shape_param_2[0]), int(shape_param_2[1])),
        (int(shape_param_2[1]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # out has shape [4096, 2048] = [A*D, B*C]. View as [A, D, B, C].
    dst = out.view(A, D, B, C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (A, D, C // BLOCK_C),
        _head_layout_clone_kernel,
        (src, dst, BLOCK_C),
    )
    return out, out.permute(1, 0)
