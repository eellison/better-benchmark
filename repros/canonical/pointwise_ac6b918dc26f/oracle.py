"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle materializes the complete XLNet view/permute/reshape/clone/view scope into the single required dense buffer and returns its contiguous `[8192,1024]` view plus transposed `[1024,8192]` alias, whereas Inductor lowers the fixed metadata chain and clone through a generic layout-copy indexer; Inductor cannot do this today because its layout algebra does not canonicalize this affine `[b,d,a,c] -> [a,b,c,d]` clone with sibling alias outputs into one tiled copy; the fix is ALGEBRAIC_ELIMINATION: recognize fixed reshape-permute-clone-view chains and emit a direct layout-copy template that writes the final dense storage once."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


A = 512
B = 16
C = 64
D = 16
OUT_ROWS = A * B
OUT_COLS = C * D


@triton.jit
def _xlnet_inverse_layout_kernel(
    in_ptr,
    out_ptr,
    BLOCK_Y: tl.constexpr,
):
    y = tl.program_id(0) * BLOCK_Y + tl.arange(0, BLOCK_Y)[:, None]
    d = tl.arange(0, 16)[None, :]

    c = y % 64
    b = (y // 64) % 16
    a = y // 1024
    mask = y < 524288

    values = tl.load(in_ptr + c + a * 64 + d * 32768 + b * 524288, mask=mask, other=0.0)
    tl.store(out_ptr + y * 16 + d, values, mask=mask)


@oracle_impl(hardware="B200", point="2cdbce9d", BLOCK_Y=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK_Y, num_warps):
    x = inputs[0]
    out = torch.empty_strided(
        (OUT_ROWS, OUT_COLS),
        (OUT_COLS, 1),
        device=x.device,
        dtype=x.dtype,
    )
    _xlnet_inverse_layout_kernel[(triton.cdiv(A * B * C, BLOCK_Y),)](
        x,
        out,
        BLOCK_Y=BLOCK_Y,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, out.t()
