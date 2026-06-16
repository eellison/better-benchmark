"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle fuses the MobileViT channels-last 2x2 patch gather, per-row layer-norm-backward reductions, bf16 side-output materialization, returned view/transpose aliases, and feature-wise reduction finalization while preserving the captured scale/divide constants for every point, whereas Inductor schedules the patch layout materialization, row reductions, bf16 side output, and feature reductions as separate work; Inductor cannot do this today because its scheduler cannot cooperatively split the feature reductions while keeping the shared patch-gather and row-reduction intermediates live across all returned aliases; the fix is COOPERATIVE_SPLIT_K: teach Inductor to lower this MobileViT patch-layernorm backward pattern as one fused cooperative split-K multi-output reduction with exact alias and cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _produce_and_partial_144_kernel(
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    out,
    partial_sum3,
    partial_sum4,
    partial_sum5,
    total_rows: tl.constexpr,
    k_tiles: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols0 = tl.arange(0, 128)
    cols1 = tl.arange(0, 16) + 128
    row_mask = rows < total_rows

    patch_row = rows // k_tiles
    k_idx = rows - patch_row * k_tiles
    n_idx = patch_row // 4
    patch = patch_row - n_idx * 4
    patch_h = patch // 2
    patch_w = patch - patch_h * 2
    h2 = k_idx // (width // 2)
    w2 = k_idx - h2 * (width // 2)
    h = h2 * 2 + patch_h
    w = w2 * 2 + patch_w

    base = n_idx * 144 * height * width + h * width * 144 + w * 144
    arg0_offsets0 = base[:, None] + cols0[None, :]
    arg0_offsets1 = base[:, None] + cols1[None, :]
    flat_offsets0 = rows[:, None] * 144 + cols0[None, :]
    flat_offsets1 = rows[:, None] * 144 + cols1[None, :]
    mask0 = row_mask[:, None]

    x0 = tl.load(arg0 + arg0_offsets0, mask=mask0, other=0.0).to(tl.float32)
    x1 = tl.load(arg0 + arg0_offsets1, mask=mask0, other=0.0).to(tl.float32)
    weight0 = tl.load(arg1 + cols0).to(tl.float32)
    weight1 = tl.load(arg1 + cols1).to(tl.float32)
    mul0 = x0 * weight0[None, :]
    mul1 = x1 * weight1[None, :]

    arg2_v0 = tl.load(arg2 + flat_offsets0, mask=mask0, other=0.0).to(tl.float32)
    arg2_v1 = tl.load(arg2 + flat_offsets1, mask=mask0, other=0.0).to(tl.float32)
    mean = tl.load(arg3 + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv_std = tl.load(arg4 + rows, mask=row_mask, other=0.0).to(tl.float32)
    mul2_0 = (arg2_v0 - mean[:, None]) * inv_std[:, None]
    mul2_1 = (arg2_v1 - mean[:, None]) * inv_std[:, None]

    sum1 = tl.sum(tl.where(mask0, mul0, 0.0), axis=1) + tl.sum(
        tl.where(mask0, mul1, 0.0), axis=1
    )
    sum2 = tl.sum(tl.where(mask0, mul0 * mul2_0, 0.0), axis=1) + tl.sum(
        tl.where(mask0, mul1 * mul2_1, 0.0), axis=1
    )

    out_f32_0 = (inv_std[:, None] / 144.0) * (
        (mul0 * 144.0 - sum1[:, None]) - mul2_0 * sum2[:, None]
    )
    out_f32_1 = (inv_std[:, None] / 144.0) * (
        (mul1 * 144.0 - sum1[:, None]) - mul2_1 * sum2[:, None]
    )
    out_bf16_0 = out_f32_0.to(tl.bfloat16)
    out_bf16_1 = out_f32_1.to(tl.bfloat16)
    tl.store(out + flat_offsets0, out_bf16_0, mask=mask0)
    tl.store(out + flat_offsets1, out_bf16_1, mask=mask0)

    partial_base = pid * 144
    tl.store(
        partial_sum3 + partial_base + cols0,
        tl.sum(tl.where(mask0, x0 * mul2_0, 0.0), axis=0),
    )
    tl.store(
        partial_sum3 + partial_base + cols1,
        tl.sum(tl.where(mask0, x1 * mul2_1, 0.0), axis=0),
    )
    tl.store(
        partial_sum4 + partial_base + cols0,
        tl.sum(tl.where(mask0, x0, 0.0), axis=0),
    )
    tl.store(
        partial_sum4 + partial_base + cols1,
        tl.sum(tl.where(mask0, x1, 0.0), axis=0),
    )
    tl.store(
        partial_sum5 + partial_base + cols0,
        tl.sum(tl.where(mask0, out_bf16_0.to(tl.float32), 0.0), axis=0),
    )
    tl.store(
        partial_sum5 + partial_base + cols1,
        tl.sum(tl.where(mask0, out_bf16_1.to(tl.float32), 0.0), axis=0),
    )


@triton.jit
def _produce_and_partial_kernel(
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    out,
    partial_sum3,
    partial_sum4,
    partial_sum5,
    total_rows: tl.constexpr,
    channels: tl.constexpr,
    k_tiles: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < total_rows
    col_mask = cols < channels
    mask = row_mask[:, None] & col_mask[None, :]

    patch_row = rows // k_tiles
    k_idx = rows - patch_row * k_tiles
    n_idx = patch_row // 4
    patch = patch_row - n_idx * 4
    patch_h = patch // 2
    patch_w = patch - patch_h * 2
    h2 = k_idx // (width // 2)
    w2 = k_idx - h2 * (width // 2)
    h = h2 * 2 + patch_h
    w = w2 * 2 + patch_w

    arg0_offsets = (
        n_idx[:, None] * channels * height * width
        + h[:, None] * width * channels
        + w[:, None] * channels
        + cols[None, :]
    )
    flat_offsets = rows[:, None] * channels + cols[None, :]

    x = tl.load(arg0 + arg0_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(arg1 + cols, mask=col_mask, other=0.0).to(tl.float32)
    mul = x * weight[None, :]

    arg2_v = tl.load(arg2 + flat_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(arg3 + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv_std = tl.load(arg4 + rows, mask=row_mask, other=0.0).to(tl.float32)
    centered = arg2_v - mean[:, None]
    mul2 = centered * inv_std[:, None]

    sum1 = tl.sum(tl.where(mask, mul, 0.0), axis=1)
    sum2 = tl.sum(tl.where(mask, mul * mul2, 0.0), axis=1)

    out_f32 = (inv_std[:, None] / 144.0) * (
        (mul * 144.0 - sum1[:, None]) - mul2 * sum2[:, None]
    )
    out_bf16 = out_f32.to(tl.bfloat16)
    tl.store(out + flat_offsets, out_bf16, mask=mask)

    feature_sum3 = tl.sum(tl.where(mask, x * mul2, 0.0), axis=0)
    feature_sum4 = tl.sum(tl.where(mask, x, 0.0), axis=0)
    feature_sum5 = tl.sum(tl.where(mask, out_bf16.to(tl.float32), 0.0), axis=0)
    partial_offsets = pid * channels + cols
    tl.store(partial_sum3 + partial_offsets, feature_sum3, mask=col_mask)
    tl.store(partial_sum4 + partial_offsets, feature_sum4, mask=col_mask)
    tl.store(partial_sum5 + partial_offsets, feature_sum5, mask=col_mask)


@triton.jit
def _partial_finalize_kernel(
    partial_sum3,
    partial_sum4,
    partial_sum5,
    out_sum3,
    out_sum4,
    out_sum5,
    num_partials: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    col_block = tl.program_id(0)
    partials = tl.arange(0, BLOCK_P)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (partials[:, None] < num_partials) & (cols[None, :] < channels)
    offsets = partials[:, None] * channels + cols[None, :]

    sum3 = tl.sum(tl.load(partial_sum3 + offsets, mask=mask, other=0.0), axis=0)
    sum4 = tl.sum(tl.load(partial_sum4 + offsets, mask=mask, other=0.0), axis=0)
    sum5 = tl.sum(tl.load(partial_sum5 + offsets, mask=mask, other=0.0), axis=0)

    col_mask = cols < channels
    tl.store(out_sum3 + cols, sum3, mask=col_mask)
    tl.store(out_sum4 + cols, sum4, mask=col_mask)
    tl.store(out_sum5 + cols, sum5.to(tl.bfloat16).to(tl.float32), mask=col_mask)


def _next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def _run(
    arg0_1,
    arg1_1,
    arg2_1,
    arg3_1,
    arg4_1,
    channels,
    k_tiles,
    height,
    width,
    block_m,
    block_c,
    final_block_c,
    num_warps,
):
    total_rows = 512 * k_tiles
    num_partials = triton.cdiv(total_rows, block_m)
    final_block_p = _next_power_of_2(num_partials)

    out = torch.empty_strided(
        (512, k_tiles, channels),
        (k_tiles * channels, channels, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial_sum3 = torch.empty((num_partials, channels), device=arg0_1.device, dtype=torch.float32)
    partial_sum4 = torch.empty((num_partials, channels), device=arg0_1.device, dtype=torch.float32)
    partial_sum5 = torch.empty((num_partials, channels), device=arg0_1.device, dtype=torch.float32)
    sum3 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    sum4 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    sum5 = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)

    if channels == 144:
        _produce_and_partial_144_kernel[(num_partials,)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            out,
            partial_sum3,
            partial_sum4,
            partial_sum5,
            total_rows,
            k_tiles,
            height,
            width,
            BLOCK_M=block_m,
            num_warps=num_warps,
        )
    else:
        _produce_and_partial_kernel[(num_partials,)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            out,
            partial_sum3,
            partial_sum4,
            partial_sum5,
            total_rows,
            channels,
            k_tiles,
            height,
            width,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    _partial_finalize_kernel[(triton.cdiv(channels, final_block_c),)](
        partial_sum3,
        partial_sum4,
        partial_sum5,
        sum3,
        sum4,
        sum5,
        num_partials,
        channels,
        BLOCK_P=final_block_p,
        BLOCK_C=final_block_c,
        num_warps=8,
    )

    flat = out.view((total_rows, channels))
    return sum3, sum4, out, flat, flat.permute(1, 0), sum5


@oracle_impl(
    hardware="B200",
    point="5afc2635",
    CHANNELS=144,
    K_TILES=256,
    HEIGHT=32,
    WIDTH=32,
    BLOCK_M=128,
    BLOCK_C=256,
    FINAL_BLOCK_C=16,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="052eef63",
    CHANNELS=192,
    K_TILES=64,
    HEIGHT=16,
    WIDTH=16,
    BLOCK_M=64,
    BLOCK_C=256,
    FINAL_BLOCK_C=16,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="9ff73aef",
    CHANNELS=240,
    K_TILES=16,
    HEIGHT=8,
    WIDTH=8,
    BLOCK_M=64,
    BLOCK_C=256,
    FINAL_BLOCK_C=16,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    CHANNELS: int,
    K_TILES: int,
    HEIGHT: int,
    WIDTH: int,
    BLOCK_M: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_ = inputs
    return _run(
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        CHANNELS,
        K_TILES,
        HEIGHT,
        WIDTH,
        BLOCK_M,
        BLOCK_C,
        FINAL_BLOCK_C,
        num_warps,
    )
