"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Mistral 32-output bf16 causal-mask construction in one Triton pointwise kernel, sharing the generated row/column predicate for all returned `[1,1,1000,1000]` outputs and materializing each distinct output tensor with exact bf16 `0.0` and `-inf` values; Inductor lowers the iota/unsqueeze/le/expand producer and thirty-two identical `where` graph-output siblings through generic pointwise scheduling, so the shared structured predicate is rebuilt across many output stores; the fix is SCHEDULER_FUSION: teach pointwise scheduling to fuse same-domain graph-output siblings with shared index-derived producers into one multi-output store kernel."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 1000
OUT_SHAPE = (1, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUT_NUMEL = SEQ * SEQ
OUTPUT_COUNT = 32


@triton.jit
def _causal_mask_32_kernel(
    out0,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    out7,
    out8,
    out9,
    out10,
    out11,
    out12,
    out13,
    out14,
    out15,
    out16,
    out17,
    out18,
    out19,
    out20,
    out21,
    out22,
    out23,
    out24,
    out25,
    out26,
    out27,
    out28,
    out29,
    out30,
    out31,
    n_elements: tl.constexpr,
    seq: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    rows = offsets // seq
    cols = offsets - rows * seq
    values = tl.where(cols <= rows, 0.0, -float("inf"))

    tl.store(out0 + offsets, values, mask=mask)
    tl.store(out1 + offsets, values, mask=mask)
    tl.store(out2 + offsets, values, mask=mask)
    tl.store(out3 + offsets, values, mask=mask)
    tl.store(out4 + offsets, values, mask=mask)
    tl.store(out5 + offsets, values, mask=mask)
    tl.store(out6 + offsets, values, mask=mask)
    tl.store(out7 + offsets, values, mask=mask)
    tl.store(out8 + offsets, values, mask=mask)
    tl.store(out9 + offsets, values, mask=mask)
    tl.store(out10 + offsets, values, mask=mask)
    tl.store(out11 + offsets, values, mask=mask)
    tl.store(out12 + offsets, values, mask=mask)
    tl.store(out13 + offsets, values, mask=mask)
    tl.store(out14 + offsets, values, mask=mask)
    tl.store(out15 + offsets, values, mask=mask)
    tl.store(out16 + offsets, values, mask=mask)
    tl.store(out17 + offsets, values, mask=mask)
    tl.store(out18 + offsets, values, mask=mask)
    tl.store(out19 + offsets, values, mask=mask)
    tl.store(out20 + offsets, values, mask=mask)
    tl.store(out21 + offsets, values, mask=mask)
    tl.store(out22 + offsets, values, mask=mask)
    tl.store(out23 + offsets, values, mask=mask)
    tl.store(out24 + offsets, values, mask=mask)
    tl.store(out25 + offsets, values, mask=mask)
    tl.store(out26 + offsets, values, mask=mask)
    tl.store(out27 + offsets, values, mask=mask)
    tl.store(out28 + offsets, values, mask=mask)
    tl.store(out29 + offsets, values, mask=mask)
    tl.store(out30 + offsets, values, mask=mask)
    tl.store(out31 + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    del inputs
    outputs = tuple(
        torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(OUTPUT_COUNT)
    )
    _causal_mask_32_kernel[(triton.cdiv(OUT_NUMEL, BLOCK),)](
        *outputs,
        n_elements=OUT_NUMEL,
        seq=SEQ,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return outputs
