"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 view/add/select scope by materializing the `[128,1,768]` add backing tensor in one Triton pointwise kernel and returning the exact `select(dim=1,index=0)` metadata view, whereas Inductor already collapses the singleton view/select around the add to equivalent pointwise memory traffic; Inductor cannot materially improve this local repro through scheduler fusion, algebraic elimination, split-K, or stencil fusion because the remaining work is the mandatory read of both bf16 inputs, bf16-rounded add output materialization, and launch/allocation overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor layout-indexing point unless broader pointwise launch or memory-system improvements move both paths."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bf16_add_kernel(
    arg0_ptr,
    arg1_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    lhs = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, lhs + rhs, mask=mask)


@oracle_impl(hardware="B200", point="3acce8ec", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0, arg1, shape = inputs
    rows = int(shape[0])
    cols = int(shape[2])
    numel = rows * cols

    add = torch.empty_strided(
        (rows, 1, cols),
        (cols, cols, 1),
        device=arg0.device,
        dtype=torch.bfloat16,
    )

    _bf16_add_kernel[(triton.cdiv(numel, BLOCK),)](
        arg0,
        arg1,
        add,
        N=numel,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return add.select(1, 0)
