"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Longformer generated-index scope, returning the contiguous `int64[18874368]` view produced by `iota(9437184) -> as_strided([96, 4, 768, 64], [98304, 16384, 64, 1]) -> clone -> view`. Inductor lowers this as a generic generated iota plus overlapping layout materialization; it cannot do this today because scheduler/codegen does not fold an affine iota source through a fixed `as_strided` clone into direct stores of the returned contiguous index map. The fix is ALGEBRAIC_ELIMINATION: derive each output element from its clone coordinates and write the final contiguous int64 tensor without materializing the source iota or interpreting the alias layout at runtime."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


DIM0 = 96
WINDOWS = 4
WINDOW = 768
HEAD_DIM = 64
OUTER = DIM0 * WINDOWS
NUMEL = OUTER * WINDOW * HEAD_DIM
OUTER_STRIDE = WINDOW * HEAD_DIM
SOURCE_STRIDE_DIM0 = 98304
SOURCE_STRIDE_WINDOW = 16384
SOURCE_STRIDE_POS = 64


@triton.jit
def _longformer_iota4_clone_kernel(
    out_ptr,
    WINDOW_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    OUTER_STRIDE_: tl.constexpr,
    SOURCE_STRIDE_DIM0_: tl.constexpr,
    SOURCE_STRIDE_WINDOW_: tl.constexpr,
    SOURCE_STRIDE_POS_: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    outer = tl.program_id(0)
    window_block = tl.program_id(1)

    dim0 = outer // 4
    window_id = outer - dim0 * 4
    pos = window_block * BLOCK_W + tl.arange(0, BLOCK_W)
    dim = tl.arange(0, HEAD_DIM_)
    mask = pos[:, None] < WINDOW_

    out_offsets = outer * OUTER_STRIDE_ + pos[:, None] * HEAD_DIM_ + dim[None, :]
    values = (
        dim0 * SOURCE_STRIDE_DIM0_
        + window_id * SOURCE_STRIDE_WINDOW_
        + pos[:, None] * SOURCE_STRIDE_POS_
        + dim[None, :]
    )
    tl.store(out_ptr + out_offsets, values.to(tl.int64), mask=mask)


@oracle_impl(
    hardware="B200",
    point="d7517139",
    BLOCK_W=64,
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
    _longformer_iota4_clone_kernel[(OUTER, triton.cdiv(WINDOW, BLOCK_W))](
        out,
        WINDOW_=WINDOW,
        HEAD_DIM_=HEAD_DIM,
        OUTER_STRIDE_=OUTER_STRIDE,
        SOURCE_STRIDE_DIM0_=SOURCE_STRIDE_DIM0,
        SOURCE_STRIDE_WINDOW_=SOURCE_STRIDE_WINDOW,
        SOURCE_STRIDE_POS_=SOURCE_STRIDE_POS,
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
