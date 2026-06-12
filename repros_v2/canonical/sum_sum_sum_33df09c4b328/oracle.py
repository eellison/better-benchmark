"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BEiT bf16 layer-norm-backward tuple by row-tiling the `[128*197, 768]` producer, sharing the hidden-dimension row reductions, writing the required fp32 add side output and bf16 scaled side output, returning the transpose as a metadata alias, and accumulating all compatible column reductions from the same row tiles with the captured final bf16 sum rounding, whereas Inductor schedules the row reductions, dependent pointwise epilogue, sibling column reductions, bf16 cast/sum boundary, and side-output materializations as separate generic kernels over intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, required materialized side outputs, dtype-rounding boundaries, and multiple dependent column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a split-row multi-output reduction template that fuses the row-summary producer with side-output stores and finalizes compatible channel accumulators together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128 * 197
CHANNELS = 768
TILE_ROWS = 8
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 16
VECTOR_BLOCK = 1024


@triton.jit
def _row_tile_kernel(
    x_ptr,
    gamma1_ptr,
    norm_ptr,
    div_ptr,
    residual_ptr,
    gamma2_ptr,
    other_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_add_other_ptr,
    out_scaled_sum_ptr,
    add_out_ptr,
    scaled_out_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tile_row = tl.program_id(0)
    rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    row_mask = rows < ROWS_
    channel_mask = channels < CHANNELS_
    mask = row_mask[:, None] & channel_mask[None, :]
    offsets = rows[:, None] * CHANNELS_ + channels[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    other = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma1 = tl.load(gamma1_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    gamma2 = tl.load(gamma2_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    div = tl.load(div_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    weighted = x * gamma1[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * norm, 0.0), axis=1)
    ln_grad = div[:, None] * (
        weighted * CHANNELS_ - row_sum[:, None] - norm * row_dot[:, None]
    )
    add_value = residual + ln_grad
    scaled = add_value * gamma2[None, :]
    scaled_bf16 = scaled.to(tl.bfloat16)

    tl.store(add_out_ptr + offsets, add_value, mask=mask)
    tl.store(scaled_out_ptr + offsets, scaled_bf16, mask=mask)

    tl.atomic_add(
        out_x_norm_ptr + channels,
        tl.sum(tl.where(mask, x * norm, 0.0), axis=0),
        sem="relaxed",
        mask=channel_mask,
    )
    tl.atomic_add(
        out_x_ptr + channels,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        sem="relaxed",
        mask=channel_mask,
    )
    tl.atomic_add(
        out_add_other_ptr + channels,
        tl.sum(tl.where(mask, add_value * other, 0.0), axis=0),
        sem="relaxed",
        mask=channel_mask,
    )
    tl.atomic_add(
        out_scaled_sum_ptr + channels,
        tl.sum(tl.where(mask, scaled_bf16.to(tl.float32), 0.0), axis=0),
        sem="relaxed",
        mask=channel_mask,
    )


@triton.jit
def _zero_outputs_kernel(
    out_x_norm_ptr,
    out_x_ptr,
    out_add_other_ptr,
    out_scaled_sum_ptr,
    CHANNELS_: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    zero = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    tl.store(out_x_norm_ptr + channels, zero, mask=channel_mask)
    tl.store(out_x_ptr + channels, zero, mask=channel_mask)
    tl.store(out_add_other_ptr + channels, zero, mask=channel_mask)
    tl.store(out_scaled_sum_ptr + channels, zero, mask=channel_mask)


@triton.jit
def _round_scaled_sum_kernel(
    out_scaled_sum_ptr,
    CHANNELS_: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    values = tl.load(out_scaled_sum_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        out_scaled_sum_ptr + channels,
        values.to(tl.bfloat16).to(tl.float32),
        mask=channel_mask,
    )


@oracle_impl(hardware="B200", point="7635d0ad", num_warps=8)
def oracle_forward(inputs, *, num_warps=8):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        shape_param_0,
        _shape_param_1,
        _shape_param_2,
        shape_param_3,
        _shape_param_4,
    ) = inputs

    device = arg0_1.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    out_x_norm = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_add_other = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_scaled_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    add_out = torch.empty_strided(
        tuple(shape_param_0),
        (197 * CHANNELS, CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    scaled_out = torch.empty_strided(
        tuple(shape_param_3),
        (CHANNELS, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _zero_outputs_kernel[(triton.cdiv(CHANNELS, VECTOR_BLOCK),)](
        out_x_norm,
        out_x,
        out_add_other,
        out_scaled_sum,
        CHANNELS_=CHANNELS,
        BLOCK_CHANNELS=VECTOR_BLOCK,
        num_warps=8,
    )

    _row_tile_kernel[(num_row_tiles,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out_x_norm,
        out_x,
        out_add_other,
        out_scaled_sum,
        add_out,
        scaled_out,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=num_warps,
    )

    _round_scaled_sum_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        out_scaled_sum,
        CHANNELS_=CHANNELS,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=num_warps,
    )

    return (
        out_x_norm,
        out_x,
        add_out,
        out_add_other,
        scaled_out,
        scaled_out.permute(1, 0),
        out_scaled_sum,
    )
