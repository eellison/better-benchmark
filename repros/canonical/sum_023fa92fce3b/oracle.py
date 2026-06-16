"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Demucs bf16 GLU-derivative cat plus channel-sum scope, including the f32 casts, sigmoid/multiply producer, the visible contiguous bf16 `[4, 2048, 362]` cat output, and the returned f32 copy of the bf16-rounded `sum([0, 2])`. Inductor lowers the shared slice/sigmoid/two-branch multiply producer, cat materialization, and sibling channel reduction as generic scheduled work over the logical cat domain; it cannot do this today because scheduler/codegen does not form one cooperative split-K materialize-plus-reduce plan that shares the sigmoid producer across both cat halves while preserving the explicit bf16 cast before the reduction. The fix is COOPERATIVE_SPLIT_K: emit both bf16 cat stores and per-channel partial sums from the same producer traversal, then finalize the rounded channel sums once."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 1024
C_TOTAL = 2048
TIME = 362
ROWS = BATCH * TIME


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _produce_partials_kernel(
    grad_ptr,
    gate_pair_ptr,
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

    grad = tl.load(grad_ptr + half_offsets, mask=mask, other=0.0).to(tl.float32)
    first_half = tl.load(gate_pair_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_pair_ptr + full_offsets + C_HALF_ * TIME_,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = _f32_div(1.0, _f32_add(libdevice.exp(_f32_sub(0.0, gate)), 1.0))
    first = _f32_mul(sigmoid, grad)
    second = _f32_mul(
        _f32_mul(_f32_mul(_f32_sub(1.0, sigmoid), sigmoid), first_half),
        grad,
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
    point="6d703def",
    TILE_R=256,
    BLOCK_R=256,
    BLOCK_C=16,
    FINAL_BLOCK_C=16,
    num_warps_produce=8,
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
    grad, gate_pair = inputs
    num_row_tiles = triton.cdiv(ROWS, TILE_R)
    block_tiles = _ceil_pow2(num_row_tiles)
    device = grad.device

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty_strided(
        (2, num_row_tiles, C_HALF),
        (num_row_tiles * C_HALF, C_HALF, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((C_TOTAL,), (1,), device=device, dtype=torch.float32)

    _produce_partials_kernel[(num_row_tiles, triton.cdiv(C_HALF, BLOCK_C))](
        grad,
        gate_pair,
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
