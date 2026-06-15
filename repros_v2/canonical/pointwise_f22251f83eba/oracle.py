"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete XLNet bf16 view/permute/squeeze/view/permute/clone/view/squeeze layout scope as a tiled transpose-copy directly into the returned contiguous `[D*16,1024]` output; whereas Inductor lowers the metadata chain and clone through a generic layout-copy schedule; Inductor cannot do this today because codegen does not recognize this fixed XLNet `[16,16,64,D] -> [D,16,16,64]` clone as a plane-tiled transpose with coalesced source loads and final contiguous rows; the fix is NEW_PATTERN: add a guarded layout-copy template for this permutation that writes the final squeezed output without generic index overhead."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUT_COLS = 1024
GROUPS = 16


@triton.jit
def _xlnet_flat_transpose_kernel(
    x_ptr,
    out_ptr,
    D: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(0) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    x = tl.program_id(1) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    mask = (y < D) & (x < 16384)
    vals = tl.load(x_ptr + x * D + y, mask=mask, other=0.0)
    tl.store(out_ptr + y * 16384 + x, vals, mask=mask)


@oracle_impl(hardware="B200", point="144bae60", YBLOCK=16, XBLOCK=128, num_warps=8)
@oracle_impl(hardware="B200", point="e95c7520", YBLOCK=16, XBLOCK=128, num_warps=8)
def oracle_forward(inputs, *, YBLOCK, XBLOCK, num_warps):
    x = inputs[0]
    d = x.shape[2]
    out = torch.empty_strided((d * GROUPS, OUT_COLS), (OUT_COLS, 1), device=x.device, dtype=x.dtype)
    _xlnet_flat_transpose_kernel[(triton.cdiv(d, YBLOCK), triton.cdiv(GROUPS * OUT_COLS, XBLOCK))](
        x,
        out,
        D=d,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
