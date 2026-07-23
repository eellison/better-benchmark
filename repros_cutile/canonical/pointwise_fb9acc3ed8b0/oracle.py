"""cuTile port of pointwise_fb9acc3ed8b0: XLNet [b0,b1,a,c] -> [a,b0,b1,c] clone.

Input is bf16 `[256, 512, 64]` where 256 = B0*B1 = 16*16 and output is
contiguous `[8192, 1024]` where 8192 = A*B0 = 512*16 and 1024 = B1*C = 16*64.
The reordering follows [b0, b1, a, c] -> [a, b0, b1, c] axis permutation.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


A = 512
B0 = 16
B1 = 16
C = 64
OUT_ROWS = A * B0
OUT_COLS = B1 * C


@ct.kernel
def _xlnet_chunk_copy_kernel(
    src,        # (B0, B1, A, C) bf16
    dst,        # (A, B0, B1, C) bf16
):
    a = ct.bid(0)
    b0 = ct.bid(1)
    b1 = ct.bid(2)
    # Load one (1,1,1,C) tile and store to reordered position
    values = ct.load(src, index=(b0, b1, a, 0), shape=(1, 1, 1, C))
    ct.store(dst, index=(a, b0, b1, 0), tile=values)


@oracle_impl(hardware="B200", point="2cdbce9d", BLOCK_Y=128)
def oracle_forward(inputs, *, BLOCK_Y):
    x = inputs[0]
    out = torch.empty_strided(
        (OUT_ROWS, OUT_COLS),
        (OUT_COLS, 1),
        device=x.device,
        dtype=x.dtype,
    )
    # x is (256, 512, 64) contiguous; view as (B0, B1, A, C) = (16,16,512,64)
    src_4d = x.view(B0, B1, A, C)
    # out is (8192, 1024) contiguous; view as (A, B0, B1, C) = (512, 16, 16, 64)
    dst_4d = out.view(A, B0, B1, C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (A, B0, B1),
        _xlnet_chunk_copy_kernel,
        (src_4d, dst_4d),
    )
    return out
