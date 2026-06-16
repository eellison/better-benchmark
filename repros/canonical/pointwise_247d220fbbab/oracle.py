"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete T5 causal-mask constant scope by materializing the two required i64 `[1024]` tensors, returning the three add-derived unsqueeze aliases, and folding `iota(1024) >= 0` to a tiny true bool base whose expanded output is metadata-only, whereas Inductor lowers the decomposed iota/add/unsqueeze/ge/expand graph through generic pointwise and view scheduling; Inductor cannot do this today because its simplifier does not prove the fixed nonnegative iota predicate while preserving the returned alias/view envelope; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate folding for nonnegative iota comparisons and return expand outputs as metadata aliases of the folded base."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _fill_iota_add_true_kernel(
    iota_ptr,
    add_ptr,
    true_base_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < 1024
    values = offsets.to(tl.int64)
    tl.store(iota_ptr + offsets, values, mask=mask)
    tl.store(add_ptr + offsets, values, mask=mask)
    tl.store(true_base_ptr + offsets, offsets >= 0, mask=mask)


# d7517139: (S([8,-1,1024,1024]))
@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (expand_shape,) = inputs
    device = torch.device("cuda", 0)
    iota = torch.empty_strided((1024,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((1024,), (1,), device=device, dtype=torch.int64)
    ge = torch.empty_strided(
        (1, 1, 1024, 1),
        (1024, 1024, 1, 1),
        device=device,
        dtype=torch.bool,
    )

    _fill_iota_add_true_kernel[(1,)](
        iota,
        add,
        ge,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )

    unsqueeze = add.unsqueeze(0)
    unsqueeze_1 = unsqueeze.unsqueeze(1)
    unsqueeze_2 = unsqueeze_1.unsqueeze(3)
    expand = ge.expand(tuple(int(dim) for dim in expand_shape))
    return iota, add, unsqueeze, unsqueeze_1, unsqueeze_2, ge, expand
