"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete five-output Gemma causal-mask scope by materializing five distinct compact bf16 [1,1,1000,1000] lower-triangular bases and returning the exact [1,8,1000,1000] zero-stride head expansions, whereas Inductor lowers the repeated iota/unsqueeze/le/where/expand sibling graph as generic pointwise work; Inductor cannot do this today because its algebraic simplifier does not canonicalize repeated shape-derived causal masks into shared predicate materialization while preserving distinct expanded output storages; the fix is ALGEBRAIC_ELIMINATION: add shape-aware causal-mask folding plus sibling output commoning that emits compact bases and metadata-only expands."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 1
BASE_HEADS = 1
HEADS = 8
SEQ = 1000
BASE_SHAPE = (BATCH, BASE_HEADS, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
EXPANDED_SHAPE = (BATCH, HEADS, SEQ, SEQ)
BASE_NUMEL = BATCH * BASE_HEADS * SEQ * SEQ


@triton.jit
def _causal_mask5_kernel(
    out0,
    out1,
    out2,
    out3,
    out4,
    n_elements: tl.constexpr,
    seq: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    col = offsets % seq
    row = (offsets // seq) % seq
    value = tl.where(col <= row, 0.0, -float("inf"))

    tl.store(out0 + offsets, value, mask=mask)
    tl.store(out1 + offsets, value, mask=mask)
    tl.store(out2 + offsets, value, mask=mask)
    tl.store(out3 + offsets, value, mask=mask)
    tl.store(out4 + offsets, value, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", block=1024, num_warps=4)
def oracle_forward(inputs, *, block: int, num_warps: int):
    del inputs
    bases = tuple(
        torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device="cuda", dtype=torch.bfloat16)
        for _ in range(5)
    )
    grid = (triton.cdiv(BASE_NUMEL, block),)
    _causal_mask5_kernel[grid](
        bases[0],
        bases[1],
        bases[2],
        bases[3],
        bases[4],
        n_elements=BASE_NUMEL,
        seq=SEQ,
        block=block,
        num_warps=num_warps,
    )
    return tuple(base.expand(EXPANDED_SHAPE) for base in bases)
