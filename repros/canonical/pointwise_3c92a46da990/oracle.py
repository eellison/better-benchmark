"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle materializes the required bf16 clone buffer directly in its final squeezed layout and returns the second result as a transpose metadata view of the same storage, whereas Inductor lowers the cast plus unsqueeze/permute/clone/view/squeeze chain as generic layout movement before exposing the returned aliasing views; Inductor cannot do this today because its pointwise output planner does not canonicalize size-one view operations around a forced contiguous clone into the clone's final logical layout; the fix is ALGEBRAIC_ELIMINATION: collapse the view-only operations, choose the clone storage order as the single materialization target, and return sibling views from that buffer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cast_clone_layout_kernel(
    input_ptr,
    output_ptr,
    TBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
):
    t = tl.program_id(0) * TBLOCK + tl.arange(0, TBLOCK)
    c = tl.program_id(1) * CBLOCK + tl.arange(0, CBLOCK)
    b = tl.program_id(2)

    load_offsets = t[:, None] * 1024 + b * 64 + c[None, :]
    load_mask = (t[:, None] < 1024) & (c[None, :] < 64)
    values = tl.load(input_ptr + load_offsets, mask=load_mask, other=0.0)

    store_offsets = c[:, None] * 16384 + b * 1024 + t[None, :]
    store_mask = (c[:, None] < 64) & (t[None, :] < 1024)
    tl.store(output_ptr + store_offsets, tl.trans(values), mask=store_mask)


@oracle_impl(hardware="B200", point="d102a86e", TBLOCK=128, CBLOCK=64, num_warps=8)
def oracle_forward(inputs, *, TBLOCK=128, CBLOCK=64, num_warps=8):
    arg0_1, shape_param = inputs
    rows = int(shape_param[1])
    cols = int(shape_param[2])

    base = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(cols, TBLOCK), triton.cdiv(64, CBLOCK), 16)
    _cast_clone_layout_kernel[grid](
        arg0_1,
        base,
        TBLOCK=TBLOCK,
        CBLOCK=CBLOCK,
        num_warps=num_warps,
    )
    return base, base.permute(1, 0)
