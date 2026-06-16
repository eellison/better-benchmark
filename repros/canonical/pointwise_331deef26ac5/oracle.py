"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete BEiT three-input `f32[768]` cat followed by bf16 conversion in one segment-aware Triton kernel, preserving the captured formulation where the f32 inputs are concatenated into a contiguous `[2304]` logical result before RTNE bf16 output rounding; Inductor lowers this small fixed-list cat/cast through generic pointwise cat indexing, which cannot emit the straight-line equal-segment loads and stores used here today; the fix is NEW_PATTERN: add a guarded small fixed-segment cat materialization template that sinks the final dtype conversion into the segment stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cat3_f32_to_bf16_kernel(
    in0,
    in1,
    in2,
    out,
    SEGMENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < SEGMENT

    values0 = tl.load(in0 + offsets, mask=mask, other=0.0)
    values1 = tl.load(in1 + offsets, mask=mask, other=0.0)
    values2 = tl.load(in2 + offsets, mask=mask, other=0.0)

    tl.store(out + offsets, values0.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    tl.store(out + SEGMENT + offsets, values1.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    tl.store(out + 2 * SEGMENT + offsets, values2.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(hardware="B200", point="61a2330b", BLOCK=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK, num_warps):
    in0, in1, in2 = inputs
    out = torch.empty_strided((2304,), (1,), device=in0.device, dtype=torch.bfloat16)
    _cat3_f32_to_bf16_kernel[(1,)](
        in0,
        in1,
        in2,
        out,
        SEGMENT=768,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
