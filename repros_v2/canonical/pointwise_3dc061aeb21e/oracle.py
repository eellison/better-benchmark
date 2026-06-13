"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Bart/MBart/PLBart causal-mask predicate scope by materializing only the tiny `[1,1,1024,1]` bool base for the captured `iota(1024) >= 0` result and returning the `[8,1,1024,1024]` expand as a metadata view, whereas Inductor lowers the decomposed iota/add/unsqueeze/ge/expand graph through generic pointwise and view scheduling; Inductor cannot do this today because its simplifier does not prove the fixed nonnegative iota predicate and plan the expanded bool output as a small true base plus zero-stride view; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate folding for nonnegative iota comparisons and preserve expand outputs as metadata-only aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _fill_true_mask_base_kernel(
    base_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < 1024
    tl.store(base_ptr + offsets, offsets >= 0, mask=mask)


# d7517139: (S([8,-1,1024,1024]))
@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (expand_shape,) = inputs
    base = torch.empty_strided(
        (1, 1, 1024, 1),
        (1024, 1024, 1, 1),
        device=torch.device("cuda", 0),
        dtype=torch.bool,
    )

    _fill_true_mask_base_kernel[(1,)](
        base,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )

    return base.expand(tuple(int(dim) for dim in expand_shape))
