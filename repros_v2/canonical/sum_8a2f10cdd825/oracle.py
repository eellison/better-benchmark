"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT bf16 masked materialization scope by writing the returned contiguous `[32768,512]` backing tensor, returning its `[512,32768]` permute view, and accumulating the bf16-rounded `[512]` column sum from the same masked producer tiles, whereas Inductor currently schedules the `where` materialization, metadata permute, and dim-0 reduction as separate generic work over the materialized tensor; Inductor cannot do this today because its scheduler does not fuse a required dense side output with a compatible sibling column reduction while preserving the returned view contract and bf16 sum cast boundary; the fix is SCHEDULER_FUSION: add a multi-output producer schedule that stores the materialized backing and emits reduction partials from the same traversal before a small finalizer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 32768
COLS = 512


@triton.jit
def _masked_store_partial_kernel(
    x_ptr,
    mask_ptr,
    fill_ptr,
    out_ptr,
    partial_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    active = (rows[:, None] < 32768) & (cols[None, :] < 512)
    offsets = rows[:, None] * 512 + cols[None, :]

    mask_values = tl.load(mask_ptr + offsets, mask=active, other=1) != 0
    x_values = tl.load(
        x_ptr + offsets,
        mask=active & ~mask_values,
        other=0.0,
    ).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    values = tl.where(mask_values, fill, x_values)
    values = tl.where(active, values, 0.0)

    tl.store(out_ptr + offsets, values, mask=active)
    tl.store(
        partial_ptr + row_tile * 512 + cols,
        tl.sum(values, axis=0),
        mask=cols < 512,
    )


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tiles = tl.arange(0, BLOCK_TILES)
    active = (tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < 512)
    partials = tl.load(
        partial_ptr + tiles[:, None] * 512 + cols[None, :],
        mask=active,
        other=0.0,
    ).to(tl.float32)
    sums = tl.sum(partials, axis=0)
    sums = sums.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, sums, mask=cols < 512)


@oracle_impl(
    hardware="B200",
    point="c5bb71a2",
    BLOCK_M=128,
    BLOCK_N=64,
    FINAL_BLOCK_N=32,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
):
    x, mask, fill, _shape0, _shape1, _shape2 = inputs
    del _shape0, _shape1, _shape2

    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((COLS,), device=x.device, dtype=torch.float32)
    num_row_tiles = triton.cdiv(ROWS, BLOCK_M)
    partial = torch.empty((num_row_tiles, COLS), device=x.device, dtype=torch.float32)

    _masked_store_partial_kernel[(num_row_tiles, triton.cdiv(COLS, BLOCK_N))](
        x,
        mask,
        fill,
        out,
        partial,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )
    _finalize_sum_kernel[(triton.cdiv(COLS, FINAL_BLOCK_N),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=4,
        num_stages=4,
    )
    return out, out.permute(1, 0), sum_out
