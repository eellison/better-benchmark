"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full twelve-output OPT causal-mask materialization by folding the all-true indexed batch mask and writing the exact bf16 0.0/-inf lower-triangular outputs directly, whereas Inductor lowers the decomposed iota/unsqueeze/index/bitwise/expand/where sibling graph as generic pointwise work; Inductor cannot do this today because its algebraic simplifier does not prove the shape-created full/index predicate is tautological or common the repeated where outputs before scheduling; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate folding and sibling mask materialization commoning for generated attention masks."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _causal_mask12_kernel(
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
    tl.store(out5 + offsets, value, mask=mask)
    tl.store(out6 + offsets, value, mask=mask)
    tl.store(out7 + offsets, value, mask=mask)
    tl.store(out8 + offsets, value, mask=mask)
    tl.store(out9 + offsets, value, mask=mask)
    tl.store(out10 + offsets, value, mask=mask)
    tl.store(out11 + offsets, value, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", block=1024, num_warps=4)
def oracle_forward(inputs, *, block: int, num_warps: int):
    del inputs
    shape = (4, 1, 2048, 2048)
    outputs = tuple(torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(12))
    n_elements = 4 * 2048 * 2048
    grid = (triton.cdiv(n_elements, block),)
    _causal_mask12_kernel[grid](
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        outputs[8],
        outputs[9],
        outputs[10],
        outputs[11],
        n_elements=n_elements,
        seq=2048,
        block=block,
        num_warps=num_warps,
    )
    return outputs
