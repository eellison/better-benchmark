"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 attention-head clone once into the fresh contiguous `[B,H,S,D]` storage and returns both requested views of that storage, whereas Inductor lowers the captured view/view/permute/expand/clone/view/permute chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this head-split clone as a reusable direct copy template with multiple aliasing outputs; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that writes the clone storage directly and preserves the returned view aliases."""

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


@oracle_impl(hardware="B200", point="631b8e39", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="bd432928", BLOCK_S=16, BLOCK_D=128, num_warps=4)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="d87997ca", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="af0c9f46", BLOCK_S=16, BLOCK_D=128, num_warps=4)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="cb7fdfdf", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="b63e0b0f", BLOCK_S=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_S=16, BLOCK_D=32, num_warps=4)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    B = _shape_param_2[0]
    H = _shape_param_2[1]
    S = _shape_param_2[2]
    D = _shape_param_2[3]

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
    return view_2, view_2.permute(0, 2, 1)
