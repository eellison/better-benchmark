"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the complete six-output PLBart bf16 causal attention-mask tuple from the shape-derived iota/le predicate in one Triton pointwise kernel, whereas Inductor lowers six identical where fills over the same expanded predicate as repeated generic pointwise work; Inductor cannot do this today because its pointwise scheduler does not group same-domain graph-output siblings with shared index-derived producers into one multi-output store kernel; the fix is SCHEDULER_FUSION: add graph-output sibling fusion for repeated generated causal masks while preserving fresh output storages, strides, dtype, and `0/-inf` values."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 1024
N_OUTPUTS = 6
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
N_ELEMENTS = BATCH * SEQ * SEQ


@triton.jit
def _causal_mask_6_kernel(
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
    out5_ptr,
    N: tl.constexpr,
    SEQ_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    row = (offsets // SEQ_) - ((offsets // (SEQ_ * SEQ_)) * SEQ_)
    col = offsets - (offsets // SEQ_) * SEQ_
    values = tl.where(col <= row, 0.0, -float("inf"))

    tl.store(out0_ptr + offsets, values, mask=mask)
    tl.store(out1_ptr + offsets, values, mask=mask)
    tl.store(out2_ptr + offsets, values, mask=mask)
    tl.store(out3_ptr + offsets, values, mask=mask)
    tl.store(out4_ptr + offsets, values, mask=mask)
    tl.store(out5_ptr + offsets, values, mask=mask)


# d7517139: shape-only PLBart causal mask, S([16,-1,1024,1024]).
@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (shape_param,) = inputs
    outputs = tuple(
        torch.empty_strided(
            (int(shape_param[0]), 1, int(shape_param[2]), int(shape_param[3])),
            OUT_STRIDE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(N_OUTPUTS)
    )
    _causal_mask_6_kernel[(triton.cdiv(N_ELEMENTS, BLOCK),)](
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        N=N_ELEMENTS,
        SEQ_=SEQ,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return outputs
