"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Longformer bf16 as_strided/view/permute/div/sum scope by materializing the scaled contiguous [8192, 768] backing tensor once, returning its transposed alias, and accumulating the hidden-dimension fp32 sum from the same bf16-rounded tile, whereas Inductor lowers the layout-only chain, scalar div materialization, returned transpose, and dependent reduction through generic scheduled regions; Inductor cannot do this today because the scheduler/codegen does not fuse alias-producing layout materialization with a visible side-output transpose and a reduction epilogue while preserving the bf16 divide and bf16-rounded sum input; the fix is SCHEDULER_FUSION: add a guarded alias-aware layout-plus-reduction template that sinks the scale into the output store and reuses the stored bf16 values for the hidden reduction."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
SCALE = 0.125


@triton.jit
def _scale_and_partial_sum_kernel(
    x_ptr,
    out_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    row_block = tl.program_id(0)
    hidden_block = tl.program_id(1)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hidden = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    offsets = rows[:, None] * HIDDEN_ + hidden[None, :]
    mask = (rows[:, None] < ROWS_) & (hidden[None, :] < HIDDEN_)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = (x * SCALE_).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + offsets, scaled, mask=mask)

    summed = tl.sum(tl.where(rows[:, None] < ROWS_, scaled.to(tl.float32), 0.0), axis=0)
    tl.store(
        partial_ptr + row_block * HIDDEN_ + hidden,
        summed,
        mask=hidden < HIDDEN_,
    )


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    HIDDEN_: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    hidden = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (tiles[:, None] < NUM_ROW_BLOCKS) & (hidden[None, :] < HIDDEN_)
    values = tl.load(
        partial_ptr + tiles[:, None] * HIDDEN_ + hidden[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    summed = tl.sum(values, axis=0)
    rounded = summed.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + hidden, rounded, mask=hidden < HIDDEN_)


# 37e882df: (T([6291456], bf16), S([96,2,512,64]), ...)
@oracle_impl(
    hardware="B200",
    point="37e882df",
    BLOCK_ROWS=128,
    BLOCK_HIDDEN=64,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_HIDDEN, num_warps):
    x, _shape0, _stride0, _shape1, _shape2, _shape3, _shape4, _shape5 = inputs

    scaled = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    num_row_blocks = triton.cdiv(ROWS, BLOCK_ROWS)
    partial = torch.empty(
        (num_row_blocks, HIDDEN),
        device=x.device,
        dtype=torch.float32,
    )
    reduced = torch.empty((HIDDEN,), device=x.device, dtype=torch.float32)

    grid = (num_row_blocks, triton.cdiv(HIDDEN, BLOCK_HIDDEN))
    _scale_and_partial_sum_kernel[grid](
        x,
        scaled,
        partial,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        SCALE_=SCALE,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        num_warps=num_warps,
    )
    _final_sum_kernel[(triton.cdiv(HIDDEN, BLOCK_HIDDEN),)](
        partial,
        reduced,
        HIDDEN_=HIDDEN,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        BLOCK_TILES=triton.next_power_of_2(num_row_blocks),
        NUM_ROW_BLOCKS=num_row_blocks,
        num_warps=1,
    )
    return scaled, scaled.permute(1, 0), reduced
