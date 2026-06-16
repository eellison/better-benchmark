"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete fp32 embedding gather plus captured multiply-by-one scope by directly materializing the contiguous `[batch, sequence, hidden]` output from the int64 token ids and embedding table, whereas Inductor lowers the isolated `aten.embedding.default(..., padding_idx)` followed by `aten.mul.Tensor(..., 1.0)` through generic embedding/pointwise scheduling for each shape; Inductor cannot fully remove this local overhead today across the recorded shapes because the neutral multiply is represented as a separate pointwise consumer of the gather result instead of a pure gather-copy output contract; the fix is ALGEBRAIC_ELIMINATION: canonicalize embedding multiplied by an exact scalar one into one embedding gather materialization while preserving output dtype, masks, and contiguous strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_D": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 2, "BLOCK_D": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4, "BLOCK_D": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 4, "BLOCK_D": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_D": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_D": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_D": 128}, num_warps=4, num_stages=3),
    ],
    key=["N_ROWS", "HIDDEN"],
)
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


@oracle_impl(hardware="B200", point="5be451a9")
@oracle_impl(hardware="B200", point="a3ebf15b")
@oracle_impl(hardware="B200", point="7f3a04f0")
@oracle_impl(hardware="B200", point="835de63c")
@oracle_impl(hardware="B200", point="e849d31e")
def oracle_forward(inputs):
    weight, indices = inputs
    batch = int(indices.shape[0])
    seq = int(indices.shape[1])
    hidden = int(weight.shape[1])
    n_rows = int(indices.numel())
    out = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=weight.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (
        triton.cdiv(n_rows, meta["BLOCK_M"]),
        triton.cdiv(hidden, meta["BLOCK_D"]),
    )
    _embedding_gather_kernel[grid](
        weight,
        indices,
        out,
        N_ROWS=n_rows,
        HIDDEN=hidden,
    )
    return out
