"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 GPT-Neo attention-output head-layout clone by writing the fresh contiguous `[B,H,S,D]` clone storage directly from the contiguous `[B*S,H*D]` projection and returning both the `[B*H,S,D]` view and its final transposed alias, whereas Inductor lowers the captured view/view/permute/cast/expand/clone/view/permute chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize the attention head split and same-dtype cast as a guarded direct clone-storage copy while preserving the returned aliasing view metadata; the fix is NEW_PATTERN: add a shape-specialized attention-head layout materialization template that writes the clone storage directly and returns the requested views."""

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


# af0c9f46: (T([4096,2048], bf16), S([32,128,2048]), S([32,128,16,128]), S([32,16,128,128]), S([512,128,128]))
@oracle_impl(
    hardware="B200",
    point="af0c9f46",
    BLOCK_S=16,
    BLOCK_D=128,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    B = int(_shape_param_2[0])
    H = int(_shape_param_2[1])
    S = int(_shape_param_2[2])
    D = int(_shape_param_2[3])

    clone = torch.empty_strided(
        tuple(_shape_param_2),
        (H * S * D, S * D, D, 1),
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
    view_2 = clone.view(tuple(_shape_param_3))
    return (view_2, view_2.permute(0, 2, 1))
