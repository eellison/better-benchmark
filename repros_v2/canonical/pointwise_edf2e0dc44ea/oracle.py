"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete captured `aten.embedding.default` scope as a shape-specialized Triton row gather for both Blenderbot/Pegasus points, preserving the int64 index reads, fp32 embedding-table loads, and fresh contiguous fp32 output, whereas Inductor lowers the isolated embedding through its generic indirect-indexing pointwise copy path; Inductor cannot do this today because it lacks a small-index-count embedding row-copy template that specializes the hidden width and maps output rows and columns directly for this layout; the fix is NEW_PATTERN: add a guarded embedding-gather lowering that tiles the dense hidden dimension for static index-vector shapes and emits the fresh output without generic pointwise indexing overhead."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _embedding_gather_kernel(
    weight_ptr,
    index_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < N_ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]

    token_ids = tl.load(index_ptr + rows, mask=row_mask, other=0)
    values = tl.load(
        weight_ptr + token_ids[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + rows[:, None] * HIDDEN + cols[None, :], values, mask=mask)


# 154a9c83: (T([128,2560], f32), T([128], i64))
@oracle_impl(hardware="B200", point="154a9c83", BLOCK_M=1, BLOCK_D=1024, num_warps=8, num_stages=3)
# 8ea03555: (T([1024,1024], f32), T([128], i64))
@oracle_impl(hardware="B200", point="8ea03555", BLOCK_M=2, BLOCK_D=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_D: int, num_warps: int, num_stages: int):
    weight, indices = inputs
    rows = int(indices.numel())
    hidden = int(weight.shape[1])
    output = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=weight.device,
        dtype=weight.dtype,
    )

    grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(hidden, BLOCK_D))
    _embedding_gather_kernel[grid](
        weight,
        indices,
        output,
        N_ROWS=rows,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
