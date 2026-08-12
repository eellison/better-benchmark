"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 attention-head transpose clone directly into the fresh contiguous `[B,H,D,S]` storage and returns both requested `[B*H,D,S]` and aliasing `[B*H,S,D]` views, whereas Inductor lowers the captured view/view/permute/permute/expand/clone/view/permute chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize the double-permute attention head split as a reusable direct transpose-copy template with multiple aliasing outputs; the fix is NEW_PATTERN: add a guarded attention-head transpose materialization lowering that writes the clone storage directly and preserves the returned view aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_layout_transpose_clone_kernel(
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
        out_ptr + bh * (D * S) + d_offsets * S + s_offsets,
        values,
        mask=mask,
    )


@oracle_impl(hardware="B200", point="631b8e39", BLOCK_S=64, BLOCK_D=64, num_warps=8)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_S=64, BLOCK_D=64, num_warps=8)
@oracle_impl(hardware="B200", point="cb7fdfdf", BLOCK_S=64, BLOCK_D=64, num_warps=8)
@oracle_impl(hardware="B200", point="b63e0b0f", BLOCK_S=64, BLOCK_D=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    B = int(_shape_param_2[0])
    H = int(_shape_param_2[1])
    D = int(_shape_param_2[2])
    S = int(_shape_param_2[3])

    view_2 = torch.empty_strided(
        tuple(_shape_param_3),
        (D * S, S, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (B * H, triton.cdiv(S, BLOCK_S))
    _head_layout_transpose_clone_kernel[grid](
        arg0_1,
        view_2,
        S,
        H,
        D,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return view_2, view_2.permute(0, 2, 1)
