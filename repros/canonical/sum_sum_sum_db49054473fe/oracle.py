"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DINOv2 bf16 layernorm-backward/projection fragment by sharing rowwise normalization reductions across the f32 `add` materialization, bf16 scaled side output, transpose-view alias, and sibling column reductions, whereas Inductor schedules the row reductions, materialization, side-output reductions, and column reductions as separate generic kernels; Inductor cannot do this today because its scheduler does not form one multi-output plan that keeps row-reduction scalars live through a dependent pointwise epilogue while accumulating compatible column reductions and preserving the returned view alias; the fix is SCHEDULER_FUSION: add a full-scope layernorm-backward reduction fusion template that materializes required outputs and accumulates all sibling reductions in the same epilogue schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
CHANNELS = 768
ROWS = BATCH * TOKENS
ROW_BLOCK = 16
ROW_C_BLOCK = 1024
TILE_ROWS = 64
TILE_COLS = 64
NUM_TILES = triton.cdiv(ROWS, TILE_ROWS)
FINAL_BLOCK_T = 4096
ADD_SHAPE = (BATCH, TOKENS, CHANNELS)
ADD_STRIDE = (TOKENS * CHANNELS, CHANNELS, 1)
SIDE_SHAPE = (ROWS, CHANNELS)
SIDE_STRIDE = (CHANNELS, 1)


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
def _row_reductions_kernel(
    x_ptr,
    gamma_ptr,
    norm_ptr,
    row_sum_ptr,
    row_dot_ptr,
    rows_total: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < rows_total) & (cols[None, :] < channels)
    offsets = rows[:, None] * channels + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=cols < channels, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weighted = _f32_mul(x, gamma[None, :])

    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, _f32_mul(weighted, norm), 0.0), axis=1)
    tl.store(row_sum_ptr + rows, row_sum, mask=rows < rows_total)
    tl.store(row_dot_ptr + rows, row_dot, mask=rows < rows_total)


@triton.jit
def _epilogue_reduce_kernel(
    x_ptr,
    gamma_ptr,
    norm_ptr,
    scale_ptr,
    residual_ptr,
    side_scale_ptr,
    side_in_ptr,
    row_sum_ptr,
    row_dot_ptr,
    add_out_ptr,
    side_out_ptr,
    partial_ptr,
    rows_total: tl.constexpr,
    channels: tl.constexpr,
    tokens: tl.constexpr,
    num_tiles: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile_r = tl.program_id(0)
    tile_c = tl.program_id(1)
    rows = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < rows_total) & (cols[None, :] < channels)
    offsets = rows[:, None] * channels + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=cols < channels, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    side_in = tl.load(side_in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    side_scale = tl.load(side_scale_ptr + cols, mask=cols < channels, other=0.0).to(tl.float32)

    batch = rows // tokens
    token = rows - batch * tokens
    scale = tl.load(scale_ptr + batch * 1376 + token, mask=rows < rows_total, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + rows, mask=rows < rows_total, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + rows, mask=rows < rows_total, other=0.0).to(tl.float32)

    weighted = _f32_mul(x, gamma[None, :])
    centered = _f32_sub(_f32_mul(weighted, 768.0), row_sum[:, None])
    centered = _f32_sub(centered, _f32_mul(norm, row_dot[:, None]))
    grad = _f32_mul(scale[:, None], centered)
    add_val = _f32_add(residual, grad)
    side_val = _round_bf16_to_f32(_f32_mul(add_val, side_scale[None, :]))

    tl.store(add_out_ptr + offsets, add_val, mask=mask)
    tl.store(side_out_ptr + offsets, side_val, mask=mask)

    valid = mask
    out0 = tl.sum(tl.where(valid, _f32_mul(x, norm), 0.0), axis=0)
    out1 = tl.sum(tl.where(valid, x, 0.0), axis=0)
    out3 = tl.sum(tl.where(valid, _f32_mul(add_val, side_in), 0.0), axis=0)
    out6 = tl.sum(tl.where(valid, side_val, 0.0), axis=0)

    partial = tile_r * channels + cols
    plane = num_tiles * channels
    c_mask = cols < channels
    tl.store(partial_ptr + partial, out0, mask=c_mask)
    tl.store(partial_ptr + plane + partial, out1, mask=c_mask)
    tl.store(partial_ptr + 2 * plane + partial, out3, mask=c_mask)
    tl.store(partial_ptr + 3 * plane + partial, out6, mask=c_mask)


@triton.jit
def _finalize_kernel(
    partial_ptr,
    out0_ptr,
    out1_ptr,
    out3_ptr,
    out6_ptr,
    channels: tl.constexpr,
    num_tiles: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    col = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_T)
    mask = tiles < num_tiles
    offsets = tiles * channels + col
    plane = num_tiles * channels

    out0 = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=0)
    out1 = tl.sum(tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0), axis=0)
    out3 = tl.sum(tl.load(partial_ptr + 2 * plane + offsets, mask=mask, other=0.0), axis=0)
    out6 = tl.sum(tl.load(partial_ptr + 3 * plane + offsets, mask=mask, other=0.0), axis=0)

    tl.store(out0_ptr + col, out0)
    tl.store(out1_ptr + col, out1)
    tl.store(out3_ptr + col, out3)
    tl.store(out6_ptr + col, _round_bf16_to_f32(out6))


@oracle_impl(hardware="B200", point="df4617bb")
def oracle_forward(inputs):
    x, gamma, norm, scale, residual, side_in, side_scale, *_shape_params = inputs

    row_sum = torch.empty((ROWS,), device=x.device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=x.device, dtype=torch.float32)
    add_out = torch.empty_strided(ADD_SHAPE, ADD_STRIDE, device=x.device, dtype=torch.float32)
    side_out = torch.empty_strided(SIDE_SHAPE, SIDE_STRIDE, device=x.device, dtype=torch.bfloat16)
    partial = torch.empty((4, NUM_TILES, CHANNELS), device=x.device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    out3 = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    out6 = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)

    _row_reductions_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        x,
        gamma,
        norm,
        row_sum,
        row_dot,
        rows_total=ROWS,
        channels=CHANNELS,
        BLOCK_R=ROW_BLOCK,
        BLOCK_C=ROW_C_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _epilogue_reduce_kernel[(NUM_TILES, triton.cdiv(CHANNELS, TILE_COLS))](
        x,
        gamma,
        norm,
        scale,
        residual,
        side_scale,
        side_in,
        row_sum,
        row_dot,
        add_out,
        side_out,
        partial,
        rows_total=ROWS,
        channels=CHANNELS,
        tokens=TOKENS,
        num_tiles=NUM_TILES,
        BLOCK_R=TILE_ROWS,
        BLOCK_C=TILE_COLS,
        num_warps=4,
        num_stages=4,
    )
    _finalize_kernel[(CHANNELS,)](
        partial,
        out0,
        out1,
        out3,
        out6,
        channels=CHANNELS,
        num_tiles=NUM_TILES,
        BLOCK_T=FINAL_BLOCK_T,
        num_warps=8,
        num_stages=4,
    )

    return out0, out1, add_out, out3, side_out, side_out.permute(1, 0), out6
