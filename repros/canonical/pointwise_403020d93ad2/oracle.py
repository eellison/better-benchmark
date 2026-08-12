"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle materializes the full contiguous bool[8,1024] result of the ordered `arg0 < 0` predicate for both bf16 and f32 Longformer points in one static Triton pointwise kernel, preserving NaN, +0.0, and -0.0 comparison semantics and the returned fresh bool layout, whereas Inductor already lowers this isolated one-op graph to the same mandatory one-read/one-write/one-launch envelope; Inductor cannot materially improve this local case through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because there is no surrounding producer/consumer and the visible output tensor must be allocated and written; the fix is BANDWIDTH_BOUND: record this as a launch/materialization floor unless broader pointwise launch overhead improvements move both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8
COLS = 1024
NUMEL = ROWS * COLS


@triton.jit
def _lt_zero_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values < 0.0, mask=mask)


@oracle_impl(hardware="B200", point="a7e138d5", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5002b631", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (arg0_1,) = inputs
    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    _lt_zero_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        arg0_1,
        out,
        N=NUMEL,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
