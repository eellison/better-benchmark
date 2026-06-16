"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 Whisper singleton causal-mask pad/slice/expand scope by materializing the four required zero scalar bases with Triton and returning their `[1,6,1,1]` expanded zero-stride views, preserving the repro's bf16 values, output strides, and four-output scope, whereas Inductor lowers the iota/unsqueeze/le/where, repeated constant_pad_nd, slice, and expand chain through generic scalar/layout scheduling; Inductor cannot do this today because its algebraic simplifier does not fold singleton causal-mask construction through pad-then-slice and repeated expand view returns; the fix is ALGEBRAIC_ELIMINATION: recognize this static singleton mask fragment and replace it with zero scalar base materialization plus expanded view returns."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BASE_SHAPE = (1, 1, 1, 1)
BASE_STRIDE = (8, 8, 8, 1)
OUT_SHAPE = (1, 6, 1, 1)


@triton.jit
def _zero_four_scalars_kernel(out0_ptr, out1_ptr, out2_ptr, out3_ptr):
    zero = tl.full((), 0.0, tl.float32).to(tl.bfloat16)
    tl.store(out0_ptr, zero)
    tl.store(out1_ptr, zero)
    tl.store(out2_ptr, zero)
    tl.store(out3_ptr, zero)


# d7517139: eight shape params producing four bf16[1,6,1,1] expanded zero masks.
@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    device = torch.device("cuda")
    out0_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    out1_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    out2_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    out3_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    _zero_four_scalars_kernel[(1,)](
        out0_base,
        out1_base,
        out2_base,
        out3_base,
        num_warps=1,
        num_stages=3,
    )
    return (
        out0_base.expand(OUT_SHAPE),
        out1_base.expand(OUT_SHAPE),
        out2_base.expand(OUT_SHAPE),
        out3_base.expand(OUT_SHAPE),
    )
