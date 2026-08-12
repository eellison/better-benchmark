"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 scale plus attention-head layout materialization by writing the fresh contiguous `[B*H, S, D]` result directly from the contiguous `[B*S, H*D]` input, whereas Inductor lowers the captured view/mul/view/permute/clone/view chain through generic pointwise and layout materialization scheduling; Inductor cannot do this today because it does not recognize the scaled attention head split transpose as a single direct-copy template with the scalar multiply folded into the materialization; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that fuses elementwise scalar transforms into the transposed clone write."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scaled_head_layout_kernel(
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


@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_S=32, BLOCK_D=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    B = int(_shape_param_0[0])
    S = int(_shape_param_0[1])
    H = int(_shape_param_1[2])
    D = int(_shape_param_1[3])

    out = torch.empty(
        tuple(_shape_param_2),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (B * H, triton.cdiv(S, BLOCK_S))
    _scaled_head_layout_kernel[grid](
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
