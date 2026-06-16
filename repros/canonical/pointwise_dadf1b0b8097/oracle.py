"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 scaled attention-head transpose clone directly into the fresh contiguous `[B,H,D,S]` storage and returns both requested `[B*H,D,S]` and aliasing `[B*H,S,D]` views, whereas Inductor lowers the captured view/view/permute/permute/mul/expand/clone/view/permute chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize the scaled double-permute attention head split as a reusable direct transpose-copy template with multiple aliasing outputs; the fix is NEW_PATTERN: add a guarded attention-head transpose materialization lowering that folds the scalar multiply into the direct clone-storage write and preserves the returned view aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scaled_head_layout_transpose_clone_kernel(
    in_ptr,
    out_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y_offsets = tl.program_id(0) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    s_offsets = tl.program_id(1) * XBLOCK + tl.arange(0, XBLOCK)[None, :]

    bh = y_offsets // D
    d = y_offsets - bh * D
    b = bh // H
    h = bh - b * H
    mask = s_offsets < S

    values = tl.load(
        in_ptr + (b * S + s_offsets) * (H * D) + h * D + d,
        mask=mask,
        other=0.0,
    )
    values = values.to(tl.float32) * 0.334370152488211
    tl.store(
        out_ptr + y_offsets * S + s_offsets,
        values,
        mask=mask,
    )


@oracle_impl(hardware="B200", point="bd432928", YBLOCK=16, XBLOCK=128, num_warps=4)
@oracle_impl(hardware="B200", point="1a8eaeba", YBLOCK=16, XBLOCK=128, num_warps=4)
@oracle_impl(hardware="B200", point="d87997ca", YBLOCK=16, XBLOCK=128, num_warps=4)
@oracle_impl(hardware="B200", point="d20f46e2", YBLOCK=16, XBLOCK=128, num_warps=4)
@oracle_impl(hardware="B200", point="b8160d07", YBLOCK=16, XBLOCK=128, num_warps=4)
@oracle_impl(hardware="B200", point="ad7b2a2c", YBLOCK=16, XBLOCK=128, num_warps=4)
@oracle_impl(hardware="B200", point="3ab46e72", YBLOCK=16, XBLOCK=128, num_warps=4)
def oracle_forward(inputs, *, YBLOCK, XBLOCK, num_warps):
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
    grid = (triton.cdiv(B * H * D, YBLOCK), triton.cdiv(S, XBLOCK))
    _scaled_head_layout_transpose_clone_kernel[grid](
        arg0_1,
        view_2,
        S,
        H,
        D,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
    )

    return view_2, view_2.permute(0, 2, 1)
