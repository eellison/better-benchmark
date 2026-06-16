"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete shape-derived bidirectional attention-mask scope by materializing only the two observable int64 iota/add vectors and the tiny bool predicate base, then returning the unsqueeze and expand outputs as metadata views, whereas Inductor lowers the captured iota/add/unsqueeze/ge/expand graph through generic pointwise and view scheduling; Inductor cannot do this today because its simplifier does not prove the fixed `iota(128) >= 0` predicate through the returned view/expand fanout and coalesce the observable aliases into a tiny base-materialization plan; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate folding and output-planning canonicalization for structured attention masks with returned metadata views."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _iota_add_ge_kernel(
    iota_ptr,
    add_ptr,
    ge_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < 128
    values = offsets.to(tl.int64)
    tl.store(iota_ptr + offsets, values, mask=mask)
    tl.store(add_ptr + offsets, values, mask=mask)
    tl.store(ge_ptr + offsets, values >= 0, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (expand_shape,) = inputs
    device = torch.device("cuda", 0)

    iota = torch.empty_strided((128,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((128,), (1,), device=device, dtype=torch.int64)
    ge = torch.empty_strided(
        (1, 1, 128, 1),
        (128, 128, 1, 1),
        device=device,
        dtype=torch.bool,
    )

    _iota_add_ge_kernel[(1,)](
        iota,
        add,
        ge,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )

    unsqueeze = torch.as_strided(add, (1, 128), (128, 1))
    unsqueeze_1 = torch.as_strided(add, (1, 1, 128), (128, 128, 1))
    unsqueeze_2 = torch.as_strided(add, (1, 1, 128, 1), (128, 128, 1, 1))
    expand = ge.expand(tuple(int(dim) for dim in expand_shape))
    return iota, add, unsqueeze, unsqueeze_1, unsqueeze_2, ge, expand
