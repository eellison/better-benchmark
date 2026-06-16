"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileViT bf16 layer-norm-backward tail by row-grouping the `[B*M,C]` source, sharing each hidden-row reduction, preserving the captured hardcoded `144` scale/divide and bf16 rounding boundary before the residual add, writing the returned bf16 `[B,M,C]` tensor and its view/transpose aliases, and cooperatively finalizing the two source column reductions plus the bf16-rounded side-output sum from common partials, whereas Inductor schedules the row reductions, dependent bf16 add materialization, view/permute alias fanout, and sibling column reductions as separate generic regions; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output reduction lowering that keeps row-local scalars, explicit bf16 casts, required aliasing side outputs, and compatible column partials in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add a MobileViT row-tiled reduction lowering that preserves literal constants and dtype boundaries while emitting the full side tensor and all column reductions together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROW_FACTOR = 144.0


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mobilevit_bwd_partials_kernel(
    x_ptr,
    weight_ptr,
    rhs_bf16_ptr,
    mean_ptr,
    inv_ptr,
    residual_bf16_ptr,
    add_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROW_FACTOR_: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < CHANNELS
    offsets = rows[:, None] * CHANNELS + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    row_factor_vec = tl.full((BLOCK_M,), ROW_FACTOR_, tl.float32)
    row_factor = tl.full((BLOCK_M, BLOCK_C), ROW_FACTOR_, tl.float32)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs_src = tl.load(rhs_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_bf16_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    weighted = x * weight[None, :]
    rhs = (rhs_src - mean[:, None]) * inv[:, None]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)

    centered = weighted * row_factor - row_sum[:, None]
    centered = centered - rhs * row_dot[:, None]
    delta = (inv / row_factor_vec)[:, None] * centered
    delta_bf16 = _round_bf16_to_f32(delta)
    add_bf16 = _round_bf16_to_f32(residual + delta_bf16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    partial_base = tile * 3 * CHANNELS + cols
    tl.store(
        partials_ptr + partial_base,
        tl.sum(tl.where(mask, x * rhs, 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partials_ptr + partial_base + CHANNELS,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partials_ptr + partial_base + 2 * CHANNELS,
        tl.sum(tl.where(mask, add_bf16.to(tl.float32), 0.0), axis=0),
        mask=col_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < CHANNELS
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * CHANNELS + cols[None, :]

    x_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + CHANNELS, mask=mask, other=0.0).to(
        tl.float32
    )

    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)


@triton.jit
def _finalize_all_partials_kernel(
    partials_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_add_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < CHANNELS
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * CHANNELS + cols[None, :]

    x_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partials_ptr + offsets + CHANNELS, mask=mask, other=0.0).to(
        tl.float32
    )
    add_sum = tl.load(partials_ptr + offsets + 2 * CHANNELS, mask=mask, other=0.0).to(
        tl.float32
    )

    tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
    tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
    tl.store(out_add_sum_ptr + cols, _round_bf16_to_f32(tl.sum(add_sum, axis=0)), mask=col_mask)


@triton.jit
def _side_sum_tile_kernel(
    partials_ptr,
    side_partials_ptr,
    CHANNELS: tl.constexpr,
    CHUNKS_PER_BATCH: tl.constexpr,
    CHUNK_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunks = tl.arange(0, CHUNK_BLOCK)
    col_mask = cols < CHANNELS
    mask = (chunks[:, None] < CHUNKS_PER_BATCH) & col_mask[None, :]
    tile = b * CHUNKS_PER_BATCH + chunks
    offsets = tile[:, None] * 3 * CHANNELS + 2 * CHANNELS + cols[None, :]
    vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(tl.where(mask, vals, 0.0), axis=0)
    tl.store(side_partials_ptr + b * CHANNELS + cols, sums, mask=col_mask)


@triton.jit
def _side_sum_b_finalize_kernel(
    side_partials_ptr,
    out_add_sum_ptr,
    BATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    b = tl.arange(0, BLOCK_B)
    col_mask = cols < CHANNELS
    mask = (b[:, None] < BATCH) & col_mask[None, :]
    offsets = b[:, None] * CHANNELS + cols[None, :]
    vals = tl.load(side_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(tl.where(mask, vals, 0.0), axis=0)
    tl.store(out_add_sum_ptr + cols, _round_bf16_to_f32(sums), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 30b03cad: (T([131072,144], bf16), T([144], f32), ..., S([512,256,144]))
@oracle_impl(hardware="B200", point="30b03cad", BLOCK_M=16, BLOCK_C=256, FINAL_BLOCK_C=8, SIDE_BLOCK_C=32, GROUPED_SIDE=False, SIDE_WARPS=1, num_warps=2)
# 1c6da2dd: (T([32768,192], bf16), T([192], f32), ..., S([512,64,192]))
@oracle_impl(hardware="B200", point="1c6da2dd", BLOCK_M=64, BLOCK_C=256, FINAL_BLOCK_C=8, SIDE_BLOCK_C=32, GROUPED_SIDE=True, SIDE_WARPS=1, num_warps=8)
# 0c9dc299: (T([8192,240], bf16), T([240], f32), ..., S([512,16,240]))
@oracle_impl(hardware="B200", point="0c9dc299", BLOCK_M=64, BLOCK_C=256, FINAL_BLOCK_C=8, SIDE_BLOCK_C=32, GROUPED_SIDE=False, SIDE_WARPS=1, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    SIDE_BLOCK_C: int,
    GROUPED_SIDE: bool,
    SIDE_WARPS: int,
    num_warps: int,
):
    (
        x_bf16,
        weight,
        rhs_bf16,
        mean,
        inv,
        residual_bf16,
        full_shape_param,
        flat_shape_param,
        sum_shape_param,
    ) = inputs
    full_shape = _shape_tuple(full_shape_param)
    flat_shape = _shape_tuple(flat_shape_param)
    sum_shape = _shape_tuple(sum_shape_param)
    rows = int(flat_shape[0])
    channels = int(flat_shape[1])
    batch = int(full_shape[0])
    m_dim = int(full_shape[1])

    add_out = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    num_groups = triton.cdiv(rows, BLOCK_M)
    partials = torch.empty_strided(
        (num_groups, 3, channels),
        (3 * channels, channels, 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    _mobilevit_bwd_partials_kernel[(num_groups,)](
        x_bf16,
        weight,
        rhs_bf16,
        mean,
        inv,
        residual_bf16,
        add_out,
        partials,
        ROWS=rows,
        CHANNELS=channels,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        ROW_FACTOR_=ROW_FACTOR,
        num_warps=num_warps,
    )

    out_x_rhs = torch.empty_strided(
        sum_shape, (1,), device=x_bf16.device, dtype=torch.float32
    )
    out_x = torch.empty_strided(
        sum_shape, (1,), device=x_bf16.device, dtype=torch.float32
    )
    out_add_sum = torch.empty_strided(
        sum_shape, (1,), device=x_bf16.device, dtype=torch.float32
    )
    group_block = 1 << (num_groups - 1).bit_length()
    if GROUPED_SIDE:
        _finalize_partials_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
            partials,
            out_x_rhs,
            out_x,
            NUM_GROUPS=num_groups,
            CHANNELS=channels,
            GROUP_BLOCK=group_block,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=8,
        )
        side_partials = torch.empty_strided(
            (batch, channels),
            (channels, 1),
            device=x_bf16.device,
            dtype=torch.float32,
        )
        chunks_per_batch = triton.cdiv(m_dim, BLOCK_M)
        _side_sum_tile_kernel[(batch, triton.cdiv(channels, SIDE_BLOCK_C))](
            partials,
            side_partials,
            CHANNELS=channels,
            CHUNKS_PER_BATCH=chunks_per_batch,
            CHUNK_BLOCK=1 << (chunks_per_batch - 1).bit_length(),
            BLOCK_C=SIDE_BLOCK_C,
            num_warps=SIDE_WARPS,
        )
        _side_sum_b_finalize_kernel[(triton.cdiv(channels, SIDE_BLOCK_C),)](
            side_partials,
            out_add_sum,
            BATCH=batch,
            CHANNELS=channels,
            BLOCK_B=1 << (batch - 1).bit_length(),
            BLOCK_C=SIDE_BLOCK_C,
            num_warps=SIDE_WARPS,
        )
    else:
        _finalize_all_partials_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
            partials,
            out_x_rhs,
            out_x,
            out_add_sum,
            NUM_GROUPS=num_groups,
            CHANNELS=channels,
            GROUP_BLOCK=group_block,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=8,
        )

    flat = add_out.view(flat_shape)
    return out_x_rhs, out_x, add_out, flat, flat.permute(1, 0), out_add_sum
