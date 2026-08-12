"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 view/convert/permute/clone/view/permute scope by directly materializing the fresh contiguous `[A,C,B,D]` clone storage from the original `[A*B,C,D]` input and returning both the `[A*C,B*D]` view and its `[B*D,A*C]` transpose view, whereas Inductor lowers the dtype-preserving convert plus layout chain through generic pointwise/layout materialization; Inductor cannot do this today because its scheduler/codegen does not recognize a same-dtype convert feeding an attention-head transpose clone with multiple aliasing returned views as one specialized copy pattern; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that elides same-dtype converts, writes the clone storage directly, and preserves the returned view aliases."""

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
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    d = offsets % D
    b = (offsets // D) % B
    c = (offsets // (B * D)) % C
    a = offsets // (C * B * D)

    values = tl.load(
        in_ptr + ((a * B + b) * C + c) * D + d,
    )
    tl.store(out_ptr + offsets, values)


@oracle_impl(hardware="B200", point="e6f344ac", BLOCK=1024, num_warps=2)
def oracle_forward(inputs, *, BLOCK, num_warps):
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
    n_elements = A * B * C * D
    grid = (triton.cdiv(n_elements, BLOCK),)
    _head_transpose_clone_kernel[grid](
        arg0_1,
        clone,
        B,
        C,
        D,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )

    view_2 = clone.view(tuple(_shape_param_2))
    return view_2, view_2.permute(1, 0)
