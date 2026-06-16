"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle collapses the complete XLNet view/permute/squeeze/view/permute/clone/view layout chain into one shape-specialized Triton materialization, whereas Inductor lowers the same required clone through a generic layout-copy indexer; Inductor cannot do this today because its layout algebra does not recognize this fixed `[a,b,c,d] -> [b,d,a,c]` clone/view pattern as a direct tiled transpose-copy with contiguous input and output slices; the fix is ALGEBRAIC_ELIMINATION: canonicalize fixed reshape-permute-clone chains to an affine layout-copy template that writes the final dense output directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


A = 512
B = 16
C = 64
D = 16
OUT_STRIDE_BD = A * C


@triton.jit
def _xlnet_layout_clone_kernel(
    in_ptr,
    out_ptr,
    B_VALUE: tl.constexpr,
    C_VALUE: tl.constexpr,
    D_VALUE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    a = tl.program_id(0)
    b = tl.program_id(1)
    c_tile = tl.program_id(2)

    c = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    d = tl.arange(0, D_VALUE)
    load_offsets = a * (B_VALUE * C_VALUE * D_VALUE) + b * (C_VALUE * D_VALUE) + c[:, None] * D_VALUE + d[None, :]
    values = tl.load(in_ptr + load_offsets, mask=c[:, None] < C_VALUE, other=0.0)

    bd = b * D_VALUE + d
    store_offsets = bd[:, None] * OUT_STRIDE + a * C_VALUE + c[None, :]
    tl.store(out_ptr + store_offsets, tl.trans(values), mask=c[None, :] < C_VALUE)


@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_C=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_C, num_warps):
    source = inputs[0]
    out = torch.empty_strided(
        (B * D, A, C),
        (A * C, C, 1),
        device=source.device,
        dtype=source.dtype,
    )
    _xlnet_layout_clone_kernel[(A, B, triton.cdiv(C, BLOCK_C))](
        source,
        out,
        B_VALUE=B,
        C_VALUE=C,
        D_VALUE=D,
        OUT_STRIDE=OUT_STRIDE_BD,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
