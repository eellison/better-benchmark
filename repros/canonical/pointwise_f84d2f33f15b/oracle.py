"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Longformer generated-index scope, returning the contiguous `int64[9437184]` view produced by `iota(6291456) -> as_strided([96, 3, 512, 64], [64, 1572864, 6144, 1]) -> clone -> view`. Inductor lowers this as generic generated iota plus overlapping layout materialization; it cannot do this today because scheduler/codegen does not fold an affine `iota` source through a fixed `as_strided` clone into direct stores of the returned contiguous index map. The fix is ALGEBRAIC_ELIMINATION: derive each output element from its clone coordinates and write the final contiguous int64 tensor without materializing the source iota or interpreting the alias layout at runtime."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUTER = 96 * 3
WINDOW = 512
HEAD_DIM = 64
NUMEL = OUTER * WINDOW * HEAD_DIM
STRIDE_OUTER = WINDOW * HEAD_DIM
STRIDE_SOURCE_BATCH = 64
STRIDE_SOURCE_OVERLAP = 1572864
STRIDE_SOURCE_WINDOW = 6144


@triton.jit
def _longformer_iota_clone_kernel(
    out_ptr,
    WINDOW_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    STRIDE_OUTER_: tl.constexpr,
    STRIDE_SOURCE_BATCH_: tl.constexpr,
    STRIDE_SOURCE_OVERLAP_: tl.constexpr,
    STRIDE_SOURCE_WINDOW_: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    outer = tl.program_id(0)
    window_block = tl.program_id(1)

    batch = outer // 3
    overlap = outer - batch * 3
    window = window_block * BLOCK_W + tl.arange(0, BLOCK_W)
    dim = tl.arange(0, HEAD_DIM_)
    mask = window[:, None] < WINDOW_

    out_offsets = outer * STRIDE_OUTER_ + window[:, None] * HEAD_DIM_ + dim[None, :]
    values = (
        batch * STRIDE_SOURCE_BATCH_
        + overlap * STRIDE_SOURCE_OVERLAP_
        + window[:, None] * STRIDE_SOURCE_WINDOW_
        + dim[None, :]
    )
    tl.store(out_ptr + out_offsets, values.to(tl.int64), mask=mask)


@oracle_impl(
    hardware="B200",
    point="d7517139",
    BLOCK_W=16,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_W: int, num_warps: int):
    del inputs
    out = torch.empty_strided(
        (NUMEL,),
        (1,),
        device=torch.device("cuda", 0),
        dtype=torch.int64,
    )
    _longformer_iota_clone_kernel[(OUTER, triton.cdiv(WINDOW, BLOCK_W))](
        out,
        WINDOW_=WINDOW,
        HEAD_DIM_=HEAD_DIM,
        STRIDE_OUTER_=STRIDE_OUTER,
        STRIDE_SOURCE_BATCH_=STRIDE_SOURCE_BATCH,
        STRIDE_SOURCE_OVERLAP_=STRIDE_SOURCE_OVERLAP,
        STRIDE_SOURCE_WINDOW_=STRIDE_SOURCE_WINDOW,
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
