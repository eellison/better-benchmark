"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Blenderbot fp32 embedding gather plus multiply-by-one scope as one Triton row-copy kernel, directly materializing the fresh contiguous `[batch, 128, 2560]` output for both captured batch sizes from the int64 token indices and fp32 embedding table, whereas Inductor already lowers this isolated embedding region to the same mandatory indexed table reads and dense output stores; Inductor cannot materially improve this repro through local scheduler fusion, algebraic elimination, or a new pointwise fusion because there is no adjacent producer or consumer and the multiply by one is already semantically a no-op; the fix is BANDWIDTH_BOUND: record this as an at-floor embedding-gather case unless broader gather throughput or launch-overhead work moves both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


HIDDEN = 2560
SEQ = 128


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_D": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 2, "BLOCK_D": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4, "BLOCK_D": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 4, "BLOCK_D": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_D": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_D": 256}, num_warps=8, num_stages=3),
    ],
    key=["n_rows", "hidden"],
)
@triton.jit
def _embedding_gather_kernel(
    weight_ptr,
    index_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < n_rows
    col_mask = cols < hidden
    mask = row_mask[:, None] & col_mask[None, :]

    token_ids = tl.load(index_ptr + rows, mask=row_mask, other=0)
    values = tl.load(
        weight_ptr + token_ids[:, None] * hidden + cols[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + rows[:, None] * hidden + cols[None, :], values, mask=mask)


@oracle_impl(hardware="B200", point="c4650a85")
@oracle_impl(hardware="B200", point="0dc67f27")
def oracle_forward(inputs):
    weight, indices = inputs
    batch = int(indices.shape[0])
    n_rows = int(indices.numel())
    output = torch.empty_strided(
        (batch, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=weight.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]), triton.cdiv(HIDDEN, meta["BLOCK_D"]))
    _embedding_gather_kernel[grid](
        weight,
        indices,
        output,
        n_rows=n_rows,
        hidden=HIDDEN,
    )
    return output
