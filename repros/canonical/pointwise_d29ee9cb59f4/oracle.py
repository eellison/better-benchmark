"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Gemma sliding-window causal-mask fanout by generating the shape-only `(key > query - 4096) & (key <= query)` predicate once and writing all thirteen distinct bf16 `[1,1,1000,1000]` 0/-inf backing tensors in one Triton pass, then returning the required `[1,8,1000,1000]` zero-head-stride expanded views; Inductor lowers the repeated iota/compare/where/expand siblings as generic pointwise output scheduling; Inductor cannot do this today because the scheduler does not coalesce identical generated mask-bias siblings into one multi-output writer while preserving non-aliasing backing storages, zero-stride view metadata, and bf16 Inf semantics; the fix is SCHEDULER_FUSION: add a guarded repeated causal-mask fanout fusion for shape-derived mask predicates."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 1000
BASE_SHAPE = (1, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUT_SHAPE = (1, 8, SEQ, SEQ)
N_ELEMENTS = SEQ * SEQ


@triton.jit
def _gemma_causal_mask13_kernel(
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
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < 1000000
    key = offsets % 1000
    query = offsets // 1000
    keep = (key > (query - 4096)) & (key <= query)
    values = tl.where(keep, 0.0, -float("inf"))

    tl.store(out0 + offsets, values, mask=valid)
    tl.store(out1 + offsets, values, mask=valid)
    tl.store(out2 + offsets, values, mask=valid)
    tl.store(out3 + offsets, values, mask=valid)
    tl.store(out4 + offsets, values, mask=valid)
    tl.store(out5 + offsets, values, mask=valid)
    tl.store(out6 + offsets, values, mask=valid)
    tl.store(out7 + offsets, values, mask=valid)
    tl.store(out8 + offsets, values, mask=valid)
    tl.store(out9 + offsets, values, mask=valid)
    tl.store(out10 + offsets, values, mask=valid)
    tl.store(out11 + offsets, values, mask=valid)
    tl.store(out12 + offsets, values, mask=valid)


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    bases = tuple(
        torch.empty_strided(
            BASE_SHAPE,
            BASE_STRIDE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(13)
    )
    grid = (triton.cdiv(N_ELEMENTS, 256),)
    _gemma_causal_mask13_kernel[grid](
        bases[0],
        bases[1],
        bases[2],
        bases[3],
        bases[4],
        bases[5],
        bases[6],
        bases[7],
        bases[8],
        bases[9],
        bases[10],
        bases[11],
        bases[12],
        BLOCK=256,
        num_warps=4,
    )
    return tuple(base.expand(OUT_SHAPE) for base in bases)
