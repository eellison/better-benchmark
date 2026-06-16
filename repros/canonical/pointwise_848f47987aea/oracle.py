"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete shape-derived bidirectional attention-mask scope by materializing only the tiny bool predicate base and returning the captured expanded mask as a metadata view, whereas Inductor lowers the iota/add/unsqueeze/ge/expand graph through generic pointwise and view scheduling; Inductor cannot do this today because its simplifier does not prove the fixed `iota(128) >= 0` predicate through the returned expand output and reduce the observable work to a tiny base-materialization plan; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate folding and output-planning canonicalization for structured attention masks so this all-true mask is lowered as one small producer plus an expand view."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _ge_base_kernel(
    ge_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < 128
    values = offsets.to(tl.int64) >= 0
    tl.store(ge_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (expand_shape,) = inputs
    ge = torch.empty_strided(
        (1, 1, 128, 1),
        (128, 128, 1, 1),
        device=torch.device("cuda", 0),
        dtype=torch.bool,
    )
    _ge_base_kernel[(1,)](
        ge,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return ge.expand(tuple(int(dim) for dim in expand_shape))
