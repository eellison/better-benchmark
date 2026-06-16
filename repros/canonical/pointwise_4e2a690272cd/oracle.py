"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 view/permute/permute/clone/view scope with one shape-specialized Triton layout materialization and returns both the fresh contiguous `[N*D, H*S]` clone storage and its aliasing transpose view, whereas Inductor lowers the captured metadata chain through generic layout-copy code; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this attention-key double-permute clone as a reusable direct materialization template with multiple aliasing outputs; the fix is NEW_PATTERN: add a guarded attention-key layout materialization lowering that writes the final clone storage directly and preserves the returned view aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _key_layout_clone_kernel(
    in_ptr,
    out_ptr,
    D: tl.constexpr,
    E: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    x_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    y_offsets = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]

    n = y_offsets // D
    d = y_offsets - n * D
    mask = x_offsets < E

    values = tl.load(
        in_ptr + n * (E * D) + x_offsets * D + d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + y_offsets * E + x_offsets,
        values,
        mask=mask,
    )


@oracle_impl(hardware="B200", point="1e7ad64a", YBLOCK=32, XBLOCK=64, num_warps=4, num_stages=1)
@oracle_impl(hardware="B200", point="beb18eeb", YBLOCK=64, XBLOCK=64, num_warps=4, num_stages=1)
def oracle_forward(inputs, *, YBLOCK, XBLOCK, num_warps, num_stages):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    N = int(_shape_param_0[0])
    H = int(_shape_param_0[1])
    S = int(_shape_param_0[2])
    D = int(_shape_param_0[3])
    rows = N * D
    cols = H * S

    view_2 = torch.empty_strided(
        tuple(_shape_param_2),
        (cols, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (triton.cdiv(cols, XBLOCK), triton.cdiv(rows, YBLOCK))
    _key_layout_clone_kernel[grid](
        arg0_1,
        view_2,
        D,
        cols,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return view_2, view_2.permute(1, 0)
