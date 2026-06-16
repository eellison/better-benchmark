"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle writes the bf16 contiguous attention-head clone backing for the returned `[M, N]` tensor, returns its transposed view alias, and accumulates the bf16-rounded column sum from the same producer tiles, whereas Inductor emits the materializing permute/view/clone layout copy and then separate reduction kernels that reread the cloned buffer; Inductor cannot do this today because its scheduler does not fuse a live materialized layout-copy output with a sibling reduction over the same producer iteration space while preserving aliasing view returns and the explicit bf16 round-trip on the sum; the fix is SCHEDULER_FUSION: add a multi-output layout-copy-plus-reduction schedule that emits the returned clone storage, alias-only transposed view, and reduction partials from one producer traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _copy_partial_sum_kernel(
    x_ptr,
    clone_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)

    rows = row_tile * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    offsets = rows[:, None] * COLS + cols[None, :]

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(clone_ptr + offsets, values, mask=mask)

    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(partial_ptr + row_tile * COLS + cols, partial, mask=cols < COLS)


@triton.jit
def _finish_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col_tile = tl.program_id(0)
    cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (cols[:, None] < COLS) & (tiles[None, :] < NUM_ROW_TILES)
    values = tl.load(
        partial_ptr + tiles[None, :] * COLS + cols[:, None],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=1)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < COLS)


@oracle_impl(
    hardware="B200",
    point="903ae292",
    ROW_BLOCK=512,
    BLOCK_COLS=32,
    FINAL_BLOCK_COLS=32,
    fused_warps=8,
    final_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="9876fbcf",
    ROW_BLOCK=512,
    BLOCK_COLS=32,
    FINAL_BLOCK_COLS=32,
    fused_warps=8,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
    fused_warps: int,
    final_warps: int,
):
    x, _shape_param_0, shape_2d, sum_shape = inputs
    rows = int(shape_2d[0])
    cols = int(shape_2d[1])
    num_row_tiles = triton.cdiv(rows, ROW_BLOCK)

    clone = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (num_row_tiles, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    _copy_partial_sum_kernel[(num_row_tiles, triton.cdiv(cols, BLOCK_COLS))](
        x,
        clone,
        partial,
        ROWS=rows,
        COLS=cols,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=fused_warps,
        num_stages=3,
    )
    _finish_sum_kernel[(triton.cdiv(cols, FINAL_BLOCK_COLS),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        COLS=cols,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
        num_stages=3,
    )

    return clone, torch.as_strided(clone, (cols, rows), (1, cols)), sum_out
