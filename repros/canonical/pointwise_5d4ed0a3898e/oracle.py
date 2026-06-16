"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 Whisper scaled attention-head layout materialization by writing the contiguous `[1,6,1500,64]` clone directly from the contiguous `[1500,384]` input with the `0.125` multiply folded into the store, whereas Inductor lowers the view/mul/view/permute/clone chain through generic pointwise/layout scheduling; Inductor cannot do this today because its scheduler does not select a guarded scaled head-split transpose materialization template that specializes the affine index map and scalar multiply for this clone-only output; the fix is NEW_PATTERN: add a scaled attention-head layout materialization lowering that fuses scalar producer arithmetic into the transposed contiguous clone write."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scaled_head_clone_kernel(
    in_ptr,
    out_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    s_offsets = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    b = bh // H
    h = bh - b * H
    mask = (s_offsets < S) & (d_offsets < D)

    values = tl.load(
        in_ptr + (b * S + s_offsets) * (H * D) + h * D + d_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + (bh * S + s_offsets) * D + d_offsets,
        values * 0.125,
        mask=mask,
    )


@oracle_impl(hardware="B200", point="da5fdead", BLOCK_S=32, BLOCK_D=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D, num_warps):
    arg0_1, shape0, shape1 = inputs
    B = int(shape0[0])
    S = int(shape0[1])
    D = int(shape1[3])
    H = arg0_1.shape[1] // D

    out = torch.empty((B, H, S, D), device=arg0_1.device, dtype=arg0_1.dtype)
    _scaled_head_clone_kernel[(B * H, triton.cdiv(S, BLOCK_S))](
        arg0_1,
        out,
        S,
        H,
        D,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out
