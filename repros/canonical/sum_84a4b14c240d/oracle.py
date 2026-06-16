"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete OPT dropout-mask bf16 materialization plus sibling column-sum scope by returning the input's metadata-only `[8192, 768]` f32 view, materializing the visible bf16 masked product once with the repro's explicit bf16 cast boundaries, returning the `[768, 8192]` transpose as an alias of that product, and accumulating the dim-0 fp32 sum from the same bf16-rounded producer before the final bf16-to-f32 round-trip; Inductor lowers the visible product, aliasing transpose, and dependent reduction as separate scheduled regions and cannot currently form one alias-aware materialize-plus-column-reduction plan while preserving the exposed bf16 output and final cast boundary; the fix is SCHEDULER_FUSION: teach reduction scheduling to sink the dropout multiply into the visible producer store, expose layout-only aliases, and reuse those bf16 producer values for the compatible column reduction."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
SCALE = 1.1111111111111112


@triton.jit
def _masked_product_partial_sum_kernel(
    x_ptr,
    mask_ptr,
    product_ptr,
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
    valid = (rows[:, None] < ROWS_) & (hidden[None, :] < HIDDEN_)

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    x_bf16 = x.to(tl.bfloat16, fp_downcast_rounding="rtne")
    keep = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1).to(tl.float32)
    scale_bf16 = (keep * SCALE_).to(tl.bfloat16, fp_downcast_rounding="rtne")
    product = (x_bf16.to(tl.float32) * scale_bf16.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(product_ptr + offsets, product, mask=valid)
    partial = tl.sum(
        tl.where(rows[:, None] < ROWS_, product.to(tl.float32), 0.0),
        axis=0,
    )
    tl.store(
        partial_ptr + row_block * HIDDEN_ + hidden,
        partial,
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
    valid = (tiles[:, None] < NUM_ROW_BLOCKS) & (hidden[None, :] < HIDDEN_)
    values = tl.load(
        partial_ptr + tiles[:, None] * HIDDEN_ + hidden[None, :],
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    summed = tl.sum(values, axis=0)
    rounded = summed.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + hidden, rounded, mask=hidden < HIDDEN_)


@oracle_impl(
    hardware="B200",
    point="0955d043",
    BLOCK_ROWS=256,
    BLOCK_HIDDEN=64,
    FINAL_BLOCK_HIDDEN=64,
    producer_warps=4,
    final_warps=1,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_HIDDEN: int,
    FINAL_BLOCK_HIDDEN: int,
    producer_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    view = arg0_1.view(ROWS, HIDDEN)
    product = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    num_row_blocks = triton.cdiv(ROWS, BLOCK_ROWS)
    partial = torch.empty_strided(
        (num_row_blocks, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    reduced = torch.empty_strided(
        (HIDDEN,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _masked_product_partial_sum_kernel[
        (num_row_blocks, triton.cdiv(HIDDEN, BLOCK_HIDDEN))
    ](
        arg0_1,
        arg1_1,
        product,
        partial,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        SCALE_=SCALE,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        num_warps=producer_warps,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_HIDDEN),)](
        partial,
        reduced,
        HIDDEN_=HIDDEN,
        BLOCK_HIDDEN=FINAL_BLOCK_HIDDEN,
        BLOCK_TILES=triton.next_power_of_2(num_row_blocks),
        NUM_ROW_BLOCKS=num_row_blocks,
        num_warps=final_warps,
        num_stages=3,
    )
    return view, product, product.permute(1, 0), reduced
