"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete f32 `sum(arg0, dim=[0, 1])` scope from Repro.forward for contiguous `[B, S, H]` tensors as a row-tiled Triton column reduction that reads the full input once, accumulates column partials, and writes the contiguous `[H]` output directly, whereas Inductor lowers this isolated large-row/small-output reduction through its generic reduction template; Inductor cannot do this today because scheduler/codegen has no dedicated static hidden-dimension column-reduction template for this exact contiguous layout and reduction extent; the fix is NEW_PATTERN: add a guarded `[B*S, H] -> [H]` f32 column-sum lowering with B200-tuned row tiles and direct contiguous output stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _zero_output_kernel(
    out_ptr,
    COLS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tl.store(out_ptr + cols, tl.zeros((BLOCK_COLS,), dtype=tl.float32), mask=cols < COLS)


@triton.jit
def _atomic_sum_rows_kernel(
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    values = tl.load(
        x_ptr + rows[:, None] * COLS + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    partial = tl.sum(values, axis=0)
    tl.atomic_add(out_ptr + cols, partial, sem="relaxed", mask=cols < COLS)


@oracle_impl(hardware="B200", point="b85aeb78", BLOCK_ROWS=256, BLOCK_COLS=32, num_warps=4)
@oracle_impl(hardware="B200", point="784a7239", BLOCK_ROWS=256, BLOCK_COLS=32, num_warps=4)
@oracle_impl(hardware="B200", point="df1f991c", BLOCK_ROWS=256, BLOCK_COLS=32, num_warps=4)
@oracle_impl(hardware="B200", point="0a855bca", BLOCK_ROWS=256, BLOCK_COLS=32, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_COLS: int, num_warps: int):
    (x,) = inputs
    rows = int(x.shape[0]) * int(x.shape[1])
    cols = int(x.shape[2])
    out = torch.empty_strided((cols,), (1,), device=x.device, dtype=torch.float32)

    _zero_output_kernel[(triton.cdiv(cols, BLOCK_COLS),)](
        out,
        COLS=cols,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=1,
        num_stages=3,
    )
    _atomic_sum_rows_kernel[(triton.cdiv(rows, BLOCK_ROWS), triton.cdiv(cols, BLOCK_COLS))](
        x,
        out,
        ROWS=rows,
        COLS=cols,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
