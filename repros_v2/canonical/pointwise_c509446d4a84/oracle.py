"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 attention projection layout scope by directly materializing the fresh contiguous `[B,H,S,D]` clone storage and returning both the `[B*H,S,D]` view and `[B*H,D,S]` permuted alias, whereas Inductor lowers the captured view/view/permute/clone/view/permute chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this attention head split transpose as a direct copy template with multiple live view aliases; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that writes the clone storage directly and preserves the returned view metadata."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_layout_clone_kernel(
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
        values,
        mask=mask,
    )


@oracle_impl(hardware="B200", point="981155f5", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_S=16, BLOCK_D=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    B = int(_shape_param_0[0])
    S = int(_shape_param_0[1])
    D = int(_shape_param_2[2])
    H = int(arg0_1.shape[1]) // D

    clone = torch.empty(
        (B, H, S, D),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (B * H, triton.cdiv(S, BLOCK_S))
    _head_layout_clone_kernel[grid](
        arg0_1,
        clone,
        S,
        H,
        D,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    view_2 = clone.view(tuple(_shape_param_2))
    return view_2, view_2.permute(0, 2, 1)
