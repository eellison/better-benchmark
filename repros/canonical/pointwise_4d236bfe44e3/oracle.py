"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the full three-input bf16[768] cat returned by Repro.forward with one Triton kernel that copies each contiguous input segment directly into the bf16[2304] output, whereas Inductor currently lowers aten.cat through generic pointwise cat indexing over the concatenated space; Inductor cannot do this today because its cat lowering has no segment-aware small-list template that emits straight-line contiguous copies for fixed equal-size inputs; the fix is NEW_PATTERN: add an Inductor cat materialization template that splits fixed contiguous cat inputs into direct segment loads/stores inside one kernel without per-element source muxing."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _cat3_bf16_768_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < 768

    values0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    values1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    values2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, values0, mask=mask)
    tl.store(out_ptr + 768 + offsets, values1, mask=mask)
    tl.store(out_ptr + 1536 + offsets, values2, mask=mask)


@oracle_impl(hardware="B200", point="3d180cc7", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    in0, in1, in2 = inputs
    out = torch.empty_strided((2304,), (1,), device=in0.device, dtype=in0.dtype)
    _cat3_bf16_768_kernel[(1,)](
        in0,
        in1,
        in2,
        out,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
