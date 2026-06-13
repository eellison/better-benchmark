"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 BEiT layer-norm-backward tail by row-tiling the `[128,197,768]` producer, preserving the bf16 input promotion, row-local hidden reductions, f32 residual-gradient expression, class-token sum, returned channels-last bf16 patch view, and bf16-rounded patch channel sum from one coordinated split-K pass, whereas Inductor schedules the view/cast, row reductions, dependent pointwise expression, residual add, slice/cast/permute/view path, and sibling channel reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no multi-output reduction template that shares row scalar reductions while emitting both f32 channel summaries and view-equivalent bf16 epilogues; the fix is COOPERATIVE_SPLIT_K: add a guarded layer-norm-backward tail template that shares the row sums, writes required bf16 patch output directly, and finalizes compatible global/class/patch reductions without materializing the intermediate f32 gradient tensor."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
PATCH_START = 1
PATCH_TOKENS = 196
PATCH_HW = 14
CHANNELS = 768
ROWS = BATCH * TOKENS
INV_CHANNELS = 1.0 / CHANNELS


@triton.jit
def _row_tile_kernel(
    x_ptr,
    weight_ptr,
    source_ptr,
    mean_ptr,
    scale_ptr,
    residual_ptr,
    partial_x_centered_ptr,
    partial_x_ptr,
    partial_cls_ptr,
    partial_patch_ptr,
    patch_out_ptr,
    ROWS_: tl.constexpr,
    TOKENS_: tl.constexpr,
    PATCH_START_: tl.constexpr,
    PATCH_TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_CHANNELS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tile_row = tl.program_id(0)
    rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    row_mask = rows < ROWS_
    channel_mask = channels < CHANNELS_
    mask = row_mask[:, None] & channel_mask[None, :]

    token = rows % TOKENS_
    batch = rows // TOKENS_
    offsets = rows[:, None] * CHANNELS_ + channels[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    centered = (source - mean[:, None]) * scale[:, None]
    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * centered, 0.0), axis=1)

    term = weighted * CHANNELS_ - row_sum[:, None]
    term = term - centered * row_dot[:, None]
    grad = (scale[:, None] * INV_CHANNELS_) * term
    add_value = residual + grad

    cls_mask = mask & (token[:, None] == 0)
    patch_mask = mask & (token[:, None] >= PATCH_START_)
    patch_index = token - PATCH_START_
    patch_offsets = (
        batch[:, None] * (PATCH_TOKENS_ * CHANNELS_)
        + patch_index[:, None] * CHANNELS_
        + channels[None, :]
    )
    patch_bf16 = add_value.to(tl.bfloat16)
    tl.store(patch_out_ptr + patch_offsets, patch_bf16, mask=patch_mask)

    partial_offsets = tile_row * CHANNELS_ + channels
    tl.store(
        partial_x_centered_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * centered, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_cls_ptr + partial_offsets,
        tl.sum(tl.where(cls_mask, add_value, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_patch_ptr + partial_offsets,
        tl.sum(tl.where(patch_mask, patch_bf16.to(tl.float32), 0.0), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_kernel(
    partial_x_centered_ptr,
    partial_x_ptr,
    partial_cls_ptr,
    partial_patch_ptr,
    out_x_centered_ptr,
    out_x_ptr,
    out_cls_ptr,
    out_patch_sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_centered = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_cls = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_patch = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)

    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_base + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_x_centered += tl.sum(
            tl.load(partial_x_centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_cls += tl.sum(
            tl.load(partial_cls_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_patch += tl.sum(
            tl.load(partial_patch_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    tl.store(out_x_centered_ptr + channels, acc_x_centered, mask=channel_mask)
    tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
    tl.store(out_cls_ptr + channels, acc_cls, mask=channel_mask)
    tl.store(out_patch_sum_ptr + channels, acc_patch.to(tl.bfloat16).to(tl.float32), mask=channel_mask)


# 49c36e1b: BEiT train LN-backward tail, [128,197,768] bf16 input, f32 reductions, bf16 patch view.
@oracle_impl(
    hardware="B200",
    point="49c36e1b",
    TILE_ROWS=16,
    TILE_CHANNELS=1024,
    FINAL_BLOCK_CHANNELS=16,
    FINAL_BLOCK_TILES=1024,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    TILE_ROWS: int,
    TILE_CHANNELS: int,
    FINAL_BLOCK_CHANNELS: int,
    FINAL_BLOCK_TILES: int,
    row_warps: int,
    final_warps: int,
):
    (
        x,
        weight,
        source,
        mean,
        scale,
        residual,
        _shape0,
        _shape1,
    ) = inputs

    device = x.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partials = torch.empty((4, num_row_tiles, CHANNELS), device=device, dtype=torch.float32)
    out_x_centered = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_cls = torch.empty((1, 1, CHANNELS), device=device, dtype=torch.float32)
    patch_out = torch.empty_strided(
        (BATCH, CHANNELS, PATCH_HW, PATCH_HW),
        (PATCH_TOKENS * CHANNELS, 1, PATCH_HW * CHANNELS, CHANNELS),
        device=device,
        dtype=torch.bfloat16,
    )
    out_patch_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_row_tiles,)](
        x,
        weight,
        source,
        mean,
        scale,
        residual,
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        patch_out,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        PATCH_START_=PATCH_START,
        PATCH_TOKENS_=PATCH_TOKENS,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=row_warps,
    )

    _finalize_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        out_x_centered,
        out_x,
        out_cls,
        out_patch_sum,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=final_warps,
    )

    return out_x_centered, out_x, out_cls, patch_out, out_patch_sum
