"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete XLNet bf16 view/permute/view/unsqueeze/permute/permute/clone/view/squeeze scope directly into the fresh contiguous `[8192, 1024]` output storage, whereas Inductor lowers the captured metadata chain through generic pointwise/layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this XLNet segment/head shuffle as a direct shape-specialized transpose-copy template with the final squeeze/view epilogue sunk into the store; the fix is NEW_PATTERN: add a guarded XLNet layout materialization lowering that writes the final contiguous output layout directly while preserving the mandatory clone boundary."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _xlnet_layout_clone_kernel(
    in_ptr,
    out_ptr,
    A: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    YNUMEL: tl.constexpr,
    XNUMEL: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]

    d = y % D
    a = (y // D) % A
    s = y // (D * A)
    mask = (y < YNUMEL) & (x < XNUMEL)

    values = tl.load(
        in_ptr + d + D * s + (S * D) * x + (B * S * D) * a,
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + y * B + x, values, mask=mask)


@oracle_impl(
    hardware="B200",
    point="2cdbce9d",
    YBLOCK=128,
    XBLOCK=16,
    num_warps=1,
    num_stages=1,
)
def oracle_forward(inputs, *, YBLOCK, XBLOCK, num_warps, num_stages):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    A = int(_shape_param_0[0])
    B = int(_shape_param_0[1])
    S = int(_shape_param_0[2])
    D = int(_shape_param_0[4])
    rows = int(_shape_param_2[1])
    cols = int(_shape_param_2[2])

    output = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    ynumel = rows * D
    grid = (triton.cdiv(B, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    _xlnet_layout_clone_kernel[grid](
        arg0_1,
        output,
        A,
        B,
        S,
        D,
        YNUMEL=ynumel,
        XNUMEL=B,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
