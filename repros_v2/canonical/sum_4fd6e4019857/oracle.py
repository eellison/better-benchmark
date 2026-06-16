"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete huge-row bf16 softmax-backward scope by splitting each `[8192,262144]` row across column tiles, cooperatively reducing the captured `grad_scalar * bf16_round(exp(x - row_max) / row_denom)` product partials, and recomputing the same rounded probability in a tiled `prims.fma(-prob, row_product_sum, grad_scalar * prob)` epilogue, whereas Inductor lowers the exp/div/bf16 round-trip producer, row sum, and full output epilogue as a generic wide-row reduction that serializes too much work inside each row program; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K template for extremely wide reductions with a dependent full-tensor epilogue and precision-preserving producer recompute; the fix is COOPERATIVE_SPLIT_K: add a row-split softmax-backward lowering that emits tiled row-sum partials and fuses row-summary finalization plus the bf16 output epilogue without full-size f32 intermediates."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 262144


@triton.jit
def _rounded_prob(x_ptr, row_max_ptr, row_denom_ptr, offsets, row, mask):
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + row).to(tl.float32)
    prob = libdevice.exp(x - row_max) / row_denom
    return prob.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _partial_sum_kernel(
    x_ptr,
    row_max_ptr,
    row_denom_ptr,
    grad_scalar_ptr,
    partial_ptr,
    COLS_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < COLS_N
    offsets = row * COLS_N + cols

    prob = _rounded_prob(x_ptr, row_max_ptr, row_denom_ptr, offsets, row, mask)
    grad = tl.load(grad_scalar_ptr).to(tl.float32)
    product = grad * prob
    partial = tl.sum(tl.where(mask, product, 0.0), axis=0)
    tl.store(partial_ptr + row * NUM_TILES + tile, partial)


@triton.jit
def _softmax_backward_epilogue_kernel(
    x_ptr,
    row_max_ptr,
    row_denom_ptr,
    grad_scalar_ptr,
    partial_ptr,
    out_ptr,
    COLS_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    PARTIAL_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)

    tiles = tl.arange(0, PARTIAL_BLOCK)
    tile_mask = tiles < NUM_TILES
    product_sum = tl.sum(
        tl.load(partial_ptr + row * NUM_TILES + tiles, mask=tile_mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )

    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < COLS_N
    offsets = row * COLS_N + cols

    prob = _rounded_prob(x_ptr, row_max_ptr, row_denom_ptr, offsets, row, mask)
    grad = tl.load(grad_scalar_ptr).to(tl.float32)
    product = grad * prob
    out = tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [-prob, product_sum, product],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(
        out_ptr + offsets,
        out.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


# 74d25999: genai SoftmaxBackward static, bf16[8192,262144].
@oracle_impl(
    hardware="B200",
    point="74d25999",
    PARTIAL_BLOCK_N=16384,
    OUTPUT_BLOCK_N=1024,
    partial_warps=8,
    output_warps=4,
)
def oracle_forward(
    inputs,
    *,
    PARTIAL_BLOCK_N: int,
    OUTPUT_BLOCK_N: int,
    partial_warps: int,
    output_warps: int,
):
    x, row_max, row_denom, grad_scalar, shape = inputs
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    out_shape = tuple(int(dim) for dim in shape)
    partial_tiles = triton.cdiv(cols, PARTIAL_BLOCK_N)
    output_tiles = triton.cdiv(cols, OUTPUT_BLOCK_N)
    partial_block = triton.next_power_of_2(partial_tiles)

    partials = torch.empty_strided(
        (rows, partial_tiles),
        (partial_tiles, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        (cols, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _partial_sum_kernel[(rows, partial_tiles)](
        x,
        row_max,
        row_denom,
        grad_scalar,
        partials,
        COLS_N=cols,
        NUM_TILES=partial_tiles,
        BLOCK_N=PARTIAL_BLOCK_N,
        num_warps=partial_warps,
        num_stages=3,
    )
    _softmax_backward_epilogue_kernel[(rows, output_tiles)](
        x,
        row_max,
        row_denom,
        grad_scalar,
        partials,
        out,
        COLS_N=cols,
        NUM_TILES=partial_tiles,
        PARTIAL_BLOCK=partial_block,
        BLOCK_N=OUTPUT_BLOCK_N,
        num_warps=output_warps,
        num_stages=3,
    )
    return out
