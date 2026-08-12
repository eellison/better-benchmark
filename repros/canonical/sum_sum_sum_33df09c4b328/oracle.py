"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BEiT bf16 layer-norm-backward tuple by row-tiling the `[128*197, 768]` producer, sharing the hidden-dimension row reductions, writing the required fp32 add side output and bf16 scaled side output, returning the transpose as a metadata alias, and accumulating all compatible column reductions from the same row tiles with the captured final bf16 sum rounding, whereas Inductor schedules the row reductions, dependent pointwise epilogue, sibling column reductions, bf16 cast/sum boundary, and side-output materializations as separate generic kernels over intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, required materialized side outputs, dtype-rounding boundaries, and multiple dependent column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a split-row multi-output reduction template that fuses the row-summary producer with side-output stores and finalizes compatible channel accumulators together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128 * 197
CHANNELS = 768
TILE_ROWS = 16
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 4
FINAL_BLOCK_TILES = 2048


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_tile_kernel(
    x_ptr,
    gamma1_ptr,
    norm_ptr,
    div_ptr,
    residual_ptr,
    gamma2_ptr,
    other_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    partial_add_other_ptr,
    partial_scaled_sum_ptr,
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
    centered = _sub_rn(
        _mul_rn(weighted, tl.full((BLOCK_ROWS, BLOCK_CHANNELS), CHANNELS_, tl.float32)),
        row_sum[:, None],
    )
    centered = _sub_rn(centered, _mul_rn(norm, row_dot[:, None]))
    ln_grad = _mul_rn(div[:, None], centered)
    add_value = _add_rn(residual, ln_grad)
    scaled_bf16 = _mul_rn(add_value, gamma2[None, :]).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(add_out_ptr + offsets, add_value, mask=mask)
    tl.store(scaled_out_ptr + offsets, scaled_bf16, mask=mask)

    partial_offsets = tile_row * CHANNELS_ + channels
    tl.store(
        partial_x_norm_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * norm, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_add_other_ptr + partial_offsets,
        tl.sum(tl.where(mask, add_value * other, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_scaled_sum_ptr + partial_offsets,
        tl.sum(tl.where(mask, scaled_bf16.to(tl.float32), 0.0), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    partial_add_other_ptr,
    partial_scaled_sum_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_add_other_ptr,
    out_scaled_sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_norm = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_add_other = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_scaled_sum = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)

    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_base + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_x_norm += tl.sum(
            tl.load(partial_x_norm_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_add_other += tl.sum(
            tl.load(partial_add_other_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_scaled_sum += tl.sum(
            tl.load(partial_scaled_sum_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_norm_ptr + channels, acc_x_norm, mask=channel_mask)
    tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
    tl.store(out_add_other_ptr + channels, acc_add_other, mask=channel_mask)
    tl.store(
        out_scaled_sum_ptr + channels,
        acc_scaled_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
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
    partials = torch.empty(
        (4, num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    vector_outputs = torch.empty((4, CHANNELS), device=device, dtype=torch.float32)
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

    _row_tile_kernel[(num_row_tiles,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        add_out,
        scaled_out,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=num_warps,
    )

    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        vector_outputs[0],
        vector_outputs[1],
        vector_outputs[2],
        vector_outputs[3],
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=num_warps,
    )

    return (
        vector_outputs[0],
        vector_outputs[1],
        add_out,
        vector_outputs[2],
        scaled_out,
        scaled_out.permute(1, 0),
        vector_outputs[3],
    )
