"""cuTile port of pointwise_000209e1748d: the XLNet layout clone/permute chain.

Ports the Triton `_xlnet_layout_clone_kernel` to cuda.tile. The kernel views
the packed input `[a=512, b=16, c=64, d=16]` and writes out `[b*d=256, a=512, c=64]`.

Because cuTile `load` needs concrete tile shapes and takes a tile-space index
(not a byte offset), we reshape the input/output to their true logical rank
and let cuTile handle strides.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


A = 512
B = 16
C = 64
D = 16


@ct.kernel
def _xlnet_layout_clone_kernel(
    src,  # (A, B, C, D) bf16
    dst,  # (B, D, A, C) bf16
    BLOCK_C: ct.Constant[int],
):
    a = ct.bid(0)
    b = ct.bid(1)
    c_tile = ct.bid(2)

    # Load tile [1, 1, BLOCK_C, D] from src at index (a, b, c_tile, 0)
    values = ct.load(src, index=(a, b, c_tile, 0), shape=(1, 1, BLOCK_C, D))
    # squeeze -> [BLOCK_C, D]
    values = ct.reshape(values, (BLOCK_C, D))
    # Transpose C and D axes to lay them out as [D, BLOCK_C]
    values_t = ct.transpose(values)

    # Store to dst at tile (b, 0, a, c_tile) with shape (1, D, 1, BLOCK_C)
    values_t = ct.reshape(values_t, (1, D, 1, BLOCK_C))
    ct.store(dst, index=(b, 0, a, c_tile), tile=values_t)


@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_C):
    source = inputs[0]
    # Repro's source is a packed bf16[8192, 1024] laid out as [A, B, C, D]:
    #   axes = [A=512, B=16, C=64, D=16]
    # (flat[a*16384 + b*1024 + c*16 + d])
    src4 = source.view(A, B, C, D)
    out = torch.empty_strided(
        (B * D, A, C),
        (A * C, C, 1),
        device=source.device,
        dtype=source.dtype,
    )
    dst4 = out.view(B, D, A, C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (A, B, (C + BLOCK_C - 1) // BLOCK_C),
        _xlnet_layout_clone_kernel,
        (src4, dst4, BLOCK_C),
    )
    return out
