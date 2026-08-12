"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete XLNet view/permute/squeeze/view/permute/clone/view/squeeze layout chain directly into the returned contiguous `[8192,1024]` buffer, whereas Inductor lowers the fixed clone through a generic pointwise layout-copy indexer; Inductor cannot do this today because codegen does not recognize this XLNet `[b0,b1,a,c] -> [a,b0,b1,c]` clone as a row-tiled contiguous-copy pattern; the fix is NEW_PATTERN: add a guarded layout-copy template for fixed reshape-permute-clone chains that writes the final dense output layout directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


A = 512
B0 = 16
B1 = 16
C = 64
OUT_ROWS = A * B0
OUT_COLS = B1 * C
TOTAL_CHUNKS = OUT_ROWS * B1


@triton.jit
def _xlnet_chunk_copy_kernel(
    in_ptr,
    out_ptr,
    BLOCK_Y: tl.constexpr,
):
    y = tl.program_id(0) * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
    c = tl.arange(0, 64)[None, :]

    b1 = y % 16
    row = y // 16
    b0 = row % 16
    a = row // 16
    mask = y < 131072

    values = tl.load(in_ptr + b0 * 524288 + b1 * 32768 + a * 64 + c, mask=mask, other=0.0)
    tl.store(out_ptr + y * 64 + c, values, mask=mask)


@oracle_impl(hardware="B200", point="2cdbce9d", BLOCK_Y=128, num_warps=8)
def oracle_forward(inputs, *, BLOCK_Y, num_warps):
    x = inputs[0]
    out = torch.empty_strided(
        (OUT_ROWS, OUT_COLS),
        (OUT_COLS, 1),
        device=x.device,
        dtype=x.dtype,
    )
    _xlnet_chunk_copy_kernel[(triton.cdiv(TOTAL_CHUNKS, BLOCK_Y),)](
        x,
        out,
        BLOCK_Y=BLOCK_Y,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
