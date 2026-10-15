"""cuTile port of pointwise_ac6b918dc26f: XLNet inverse layout clone.

Input logically (B=16, D=16, A=512, C=64) contiguous -> output (A=512, B=16, C=64, D=16).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


A = 512
B = 16
C = 64
D = 16
OUT_ROWS = A * B
OUT_COLS = C * D


@ct.kernel
def _xlnet_inverse_layout_kernel(
    src,   # (B, D, A, C) bf16
    dst,   # (A, B, C, D) bf16
    BLOCK_C: ct.Constant[int],
    D_: ct.Constant[int],
):
    a = ct.bid(0)
    b = ct.bid(1)
    c_tile = ct.bid(2)
    # Load tile [1, D, 1, BLOCK_C] from src at (b, 0, a, c_tile)
    values = ct.load(src, index=(b, 0, a, c_tile), shape=(1, D_, 1, BLOCK_C))
    values = ct.reshape(values, (D_, BLOCK_C))
    values_t = ct.transpose(values)  # (BLOCK_C, D)
    values_t = ct.reshape(values_t, (1, 1, BLOCK_C, D_))
    ct.store(dst, index=(a, b, c_tile, 0), tile=values_t)


@oracle_impl(hardware="B200", point="2cdbce9d", BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_C):
    x = inputs[0]
    # x is contiguous [256, 512, 64] = viewed as (B=16, D=16, A=512, C=64)
    src4 = x.view(B, D, A, C)
    out = torch.empty_strided(
        (OUT_ROWS, OUT_COLS),
        (OUT_COLS, 1),
        device=x.device,
        dtype=x.dtype,
    )
    # Output (A, B, C, D) contiguous
    dst4 = out.view(A, B, C, D)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (A, B, (C + BLOCK_C - 1) // BLOCK_C),
        _xlnet_inverse_layout_kernel,
        (src4, dst4, BLOCK_C, D),
    )
    return out, out.t()
