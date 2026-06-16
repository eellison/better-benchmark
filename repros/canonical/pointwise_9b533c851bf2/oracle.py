"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 attention-head layout scope by directly materializing the fresh contiguous `[A,C,B,D]` clone storage from the contiguous `[A*B,C,D]` input and returning both the `[A*C,B*D]` view and its `[B*D,A*C]` transpose view, whereas Inductor lowers the captured view/permute/clone/view/permute chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this head-split transpose clone with multiple aliasing returned views as one specialized copy pattern, so the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that writes the clone storage directly and preserves the returned view aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_transpose_clone_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    ab = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    a = ab // B
    b = ab - a * B
    mask = (c_offsets < C) & (d_offsets < D)

    values = tl.load(
        in_ptr + (ab * C + c_offsets) * D + d_offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + ((a * C + c_offsets) * B + b) * D + d_offsets,
        values,
        mask=mask,
    )


@oracle_impl(hardware="B200", point="c6cb1dd8", BLOCK_C=8, BLOCK_D=256, num_warps=8)
@oracle_impl(hardware="B200", point="226fbbfa", BLOCK_C=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="e6f344ac", BLOCK_C=8, BLOCK_D=128, num_warps=4)
@oracle_impl(hardware="B200", point="fb089404", BLOCK_C=16, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="6c3c2efc", BLOCK_C=16, BLOCK_D=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_D, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    A = _shape_param_0[0]
    B = _shape_param_0[1]
    C = _shape_param_0[2]
    D = _shape_param_0[3]

    clone = torch.empty_strided(
        (A, C, B, D),
        (C * B * D, B * D, D, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (A * B, triton.cdiv(C, BLOCK_C))
    _head_transpose_clone_kernel[grid](
        arg0_1,
        clone,
        B,
        C,
        D,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    view_2 = clone.view(tuple(_shape_param_2))
    return view_2, view_2.permute(1, 0)
