"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16 view/permute/squeeze/clone/view layout materialization on the fixed row/channel surface `y=96*3*512, x=64`, writing the required fresh flat contiguous output directly, whereas Inductor lowers the captured metadata chain plus clone through generic layout-copy indexing; Inductor cannot do this today because its scheduler/codegen does not recognize this singleton-dimension Longformer attention layout as a reusable guarded layout-copy template with fixed row/channel tiling; the fix is NEW_PATTERN: add a layout materialization template for `view(96,3,64,512,1).permute(...).squeeze(...).contiguous().view(-1)` that writes the final clone storage directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _longformer_layout_kernel(
    input_ptr,
    output_ptr,
    YNUMEL: tl.constexpr,
    XNUMEL: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    mask = (y < YNUMEL) & (x < XNUMEL)

    seq = y % 512
    outer = y // 512

    values = tl.load(
        input_ptr + seq + 512 * x + 32768 * outer,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + x + 64 * y, values, mask=mask)


# 8c9b0625: (T([288,64,512], bf16), S([96,3,64,512,1]), S([9437184]))
@oracle_impl(
    hardware="B200",
    point="8c9b0625",
    YBLOCK=128,
    XBLOCK=64,
    num_warps=1,
    num_stages=1,
)
def oracle_forward(inputs, *, YBLOCK: int, XBLOCK: int, num_warps: int, num_stages: int):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    ynumel = int(_shape_param_0[0]) * int(_shape_param_0[1]) * int(_shape_param_0[3])
    xnumel = int(_shape_param_0[2])
    output_numel = int(_shape_param_1[0])

    output = torch.empty_strided(
        (output_numel,),
        (1,),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    _longformer_layout_kernel[grid](
        arg0_1,
        output,
        YNUMEL=ynumel,
        XNUMEL=xnumel,
        YBLOCK=YBLOCK,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
