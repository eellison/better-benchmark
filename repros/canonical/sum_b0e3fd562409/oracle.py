"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Demucs bf16 GLU cat-and-sum captured scope, including the visible contiguous bf16 `[4,1024,1450]` cat output and the returned f32 copy of the bf16 `sum([0, 2])`, whereas Inductor lowers the slice/sigmoid/two-branch multiply/cat materialization and channel reduction as generic producer and reduction regions over the logical cat domain. Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K template that shares the sigmoid producer across both cat halves while preserving the required bf16 materialization boundary and the sibling channel reduction. The fix is COOPERATIVE_SPLIT_K: fuse the two GLU branch stores with per-channel split-K partials from the rounded bf16 values, then finalize the concatenated channel sums with the same bf16 output rounding."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 512
C_TOTAL = 1024
TIME = 1450
ROWS = BATCH * TIME


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _produce_partials_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    C_HALF_: tl.constexpr,
    C_TOTAL_: tl.constexpr,
    TIME_: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.arange(0, BLOCK_R)
    rows = tl.program_id(0) * TILE_R + row_block
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = (row_block < TILE_R) & (rows < ROWS_)
    channel_mask = channels < C_HALF_
    mask = row_mask[:, None] & channel_mask[None, :]

    batch = rows // TIME_
    time = rows - batch * TIME_

    half_offsets = (
        batch[:, None] * (C_HALF_ * TIME_)
        + channels[None, :] * TIME_
        + time[:, None]
    )
    full_offsets = (
        batch[:, None] * (C_TOTAL_ * TIME_)
        + channels[None, :] * TIME_
        + time[:, None]
    )

    x = tl.load(x_ptr + half_offsets, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(y_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        y_ptr + full_offsets + C_HALF_ * TIME_,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    first = _f32_mul(sigmoid, x)
    second = _f32_mul(
        _f32_mul(_f32_mul(_f32_sub(1.0, sigmoid), sigmoid), value),
        x,
    )
    first_bf16 = first.to(tl.bfloat16, fp_downcast_rounding="rtne")
    second_bf16 = second.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + full_offsets, first_bf16, mask=mask)
    tl.store(out_ptr + full_offsets + C_HALF_ * TIME_, second_bf16, mask=mask)

    partial_offsets = tl.program_id(0) * C_HALF_ + channels
    tl.store(
        partials_ptr + partial_offsets,
        tl.sum(tl.where(mask, first_bf16.to(tl.float32), 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partials_ptr + NUM_ROW_TILES * C_HALF_ + partial_offsets,
        tl.sum(tl.where(mask, second_bf16.to(tl.float32), 0.0), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_kernel(
    partials_ptr,
    sum_out_ptr,
    C_HALF_: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (tiles[:, None] < NUM_ROW_TILES) & (channels[None, :] < C_HALF_)
    offsets = tiles[:, None] * C_HALF_ + channels[None, :]

    first = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(
        partials_ptr + NUM_ROW_TILES * C_HALF_ + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    channel_mask = channels < C_HALF_
    first_sum = tl.sum(first, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    second_sum = tl.sum(second, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_out_ptr + channels, first_sum, mask=channel_mask)
    tl.store(sum_out_ptr + C_HALF_ + channels, second_sum, mask=channel_mask)


@oracle_impl(
    hardware="B200",
    point="7b01906c",
    TILE_R=128,
    BLOCK_R=128,
    BLOCK_C=16,
    FINAL_BLOCK_C=16,
    num_warps_produce=4,
    num_warps_final=4,
)
def oracle_forward(
    inputs,
    *,
    TILE_R: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps_produce: int,
    num_warps_final: int,
):
    arg0_1, arg1_1 = inputs
    num_row_tiles = triton.cdiv(ROWS, TILE_R)
    block_tiles = _ceil_pow2(num_row_tiles)
    device = arg0_1.device

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (2, num_row_tiles, C_HALF),
        device=device,
        dtype=torch.float32,
    )
    sum_out = torch.empty((C_TOTAL,), device=device, dtype=torch.float32)

    _produce_partials_kernel[(num_row_tiles, triton.cdiv(C_HALF, BLOCK_C))](
        arg0_1,
        arg1_1,
        out,
        partials,
        ROWS_=ROWS,
        C_HALF_=C_HALF,
        C_TOTAL_=C_TOTAL,
        TIME_=TIME,
        NUM_ROW_TILES=num_row_tiles,
        TILE_R=TILE_R,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_produce,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(C_HALF, FINAL_BLOCK_C),)](
        partials,
        sum_out,
        C_HALF_=C_HALF,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_TILES=block_tiles,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )
    return out, sum_out
