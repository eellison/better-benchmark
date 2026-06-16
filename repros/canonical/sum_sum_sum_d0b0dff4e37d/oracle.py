"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 BEiT LayerNorm-backward/projection fragment by first sharing the row-local hidden reductions, then row-tiling the dependent f32 add output, bf16 projection output, transpose alias, and all four returned hidden-column reductions from common partials, whereas Inductor schedules the row reductions, internally rebuilt normalized producer, dependent dense epilogues, bf16 cast/view fanout, transpose alias, and sibling column reductions as separate generic regions; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that keeps row-local summaries, explicit bf16 cast boundaries, visible dense side outputs, and compatible column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible LayerNorm-backward column reductions across row tiles while sinking the f32/bf16 side-output stores and alias-only transpose return into the same producer/finalizer schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 25216
BATCH = 128
TOKENS = 197
HIDDEN = 768
ROW_REDUCE_ROWS = 16
TILE_ROWS = 64
TILE_COLS = 64
REDUCE_BLOCKS = triton.cdiv(ROWS, TILE_ROWS)
FINAL_BLOCKS = 512


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
def _row_reductions_kernel(
    x_ptr,
    gamma_ptr,
    proj_ptr,
    proj_weight_ptr,
    residual_norm_ptr,
    norm_mean_ptr,
    norm_scale_ptr,
    row_sum_ptr,
    row_dot_ptr,
    rows_total: tl.constexpr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < rows_total) & (cols[None, :] < hidden)
    offsets = rows[:, None] * hidden + cols[None, :]
    batch = rows // tokens
    token = rows - batch * tokens
    row_offsets = batch * tokens + token

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
    proj = tl.load(proj_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    proj_weight = tl.load(proj_weight_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
    residual_norm = tl.load(residual_norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(norm_mean_ptr + row_offsets, mask=rows < rows_total, other=0.0).to(tl.float32)
    scale = tl.load(norm_scale_ptr + row_offsets, mask=rows < rows_total, other=0.0).to(tl.float32)

    norm = _f32_add(residual_norm, _f32_mul(proj_weight[None, :], proj))
    norm = _f32_sub(norm, mean[:, None])
    norm = _f32_mul(norm, scale[:, None])
    weighted = _f32_mul(x, gamma[None, :])
    weighted_norm = _f32_mul(weighted, norm)

    tl.store(row_sum_ptr + rows, tl.sum(tl.where(mask, weighted, 0.0), axis=1), mask=rows < rows_total)
    tl.store(row_dot_ptr + rows, tl.sum(tl.where(mask, weighted_norm, 0.0), axis=1), mask=rows < rows_total)


@triton.jit
def _materialize_and_reduce_kernel(
    x_ptr,
    gamma_ptr,
    proj_ptr,
    proj_weight_ptr,
    residual_norm_ptr,
    norm_mean_ptr,
    norm_scale_ptr,
    residual_ptr,
    row_sum_ptr,
    row_dot_ptr,
    add_out_ptr,
    bf16_out_ptr,
    partial_ptr,
    rows_total: tl.constexpr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    reduce_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_c = tl.program_id(1)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < rows_total) & (cols[None, :] < hidden)
    offsets = rows[:, None] * hidden + cols[None, :]
    batch = rows // tokens
    token = rows - batch * tokens
    row_offsets = batch * tokens + token

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
    proj = tl.load(proj_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    proj_weight = tl.load(proj_weight_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
    residual_norm = tl.load(residual_norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(norm_mean_ptr + row_offsets, mask=rows < rows_total, other=0.0).to(tl.float32)
    scale = tl.load(norm_scale_ptr + row_offsets, mask=rows < rows_total, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + rows, mask=rows < rows_total, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + rows, mask=rows < rows_total, other=0.0).to(tl.float32)

    norm = _f32_add(residual_norm, _f32_mul(proj_weight[None, :], proj))
    norm = _f32_sub(norm, mean[:, None])
    norm = _f32_mul(norm, scale[:, None])
    weighted = _f32_mul(x, gamma[None, :])
    centered = _f32_sub(_f32_mul(weighted, 768.0), row_sum[:, None])
    centered = _f32_sub(centered, _f32_mul(norm, row_dot[:, None]))
    update = _f32_mul(scale[:, None] / 768.0, centered)
    add_out = _f32_add(residual, update)
    bf16_val = _f32_mul(add_out, proj_weight[None, :]).to(tl.bfloat16)
    bf16_as_f32 = bf16_val.to(tl.float32)

    tl.store(add_out_ptr + offsets, add_out, mask=mask)
    tl.store(bf16_out_ptr + offsets, bf16_val, mask=mask)

    out0_elem = _f32_mul(x, norm)
    out1_elem = x
    out3_elem = _f32_mul(add_out, proj)
    col_mask = cols < hidden
    partial_base = tile_m * hidden + cols
    stride = reduce_blocks * hidden
    tl.store(partial_ptr + partial_base, tl.sum(tl.where(mask, out0_elem, 0.0), axis=0), mask=col_mask)
    tl.store(partial_ptr + stride + partial_base, tl.sum(tl.where(mask, out1_elem, 0.0), axis=0), mask=col_mask)
    tl.store(partial_ptr + 2 * stride + partial_base, tl.sum(tl.where(mask, out3_elem, 0.0), axis=0), mask=col_mask)
    tl.store(partial_ptr + 3 * stride + partial_base, tl.sum(tl.where(mask, bf16_as_f32, 0.0), axis=0), mask=col_mask)


@triton.jit
def _finalize_reductions_kernel(
    partial_ptr,
    out0_ptr,
    out1_ptr,
    out3_ptr,
    out6_ptr,
    hidden: tl.constexpr,
    reduce_blocks: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    col = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_T)
    mask = tiles < reduce_blocks
    offsets = tiles * hidden + col
    stride = reduce_blocks * hidden

    acc0 = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    acc1 = tl.sum(tl.load(partial_ptr + stride + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    acc3 = tl.sum(tl.load(partial_ptr + 2 * stride + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    acc6 = tl.sum(tl.load(partial_ptr + 3 * stride + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    tl.store(out0_ptr + col, acc0)
    tl.store(out1_ptr + col, acc1)
    tl.store(out3_ptr + col, acc3)
    tl.store(out6_ptr + col, acc6.to(tl.bfloat16).to(tl.float32))


@oracle_impl(hardware="B200", point="7c1570d5")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs
    device = arg0_1.device

    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    partial = torch.empty((4, REDUCE_BLOCKS, HIDDEN), device=device, dtype=torch.float32)
    out0 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (BATCH, TOKENS, HIDDEN),
        (TOKENS * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    out3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    bf16_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out6 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    _row_reductions_kernel[(triton.cdiv(ROWS, ROW_REDUCE_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        row_sum,
        row_dot,
        rows_total=ROWS,
        tokens=TOKENS,
        hidden=HIDDEN,
        BLOCK_R=ROW_REDUCE_ROWS,
        BLOCK_C=1024,
        num_warps=8,
    )
    _materialize_and_reduce_kernel[(REDUCE_BLOCKS, triton.cdiv(HIDDEN, TILE_COLS))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        row_sum,
        row_dot,
        add_out,
        bf16_out,
        partial,
        rows_total=ROWS,
        tokens=TOKENS,
        hidden=HIDDEN,
        reduce_blocks=REDUCE_BLOCKS,
        BLOCK_M=TILE_ROWS,
        BLOCK_C=TILE_COLS,
        num_warps=4,
    )
    _finalize_reductions_kernel[(HIDDEN,)](
        partial,
        out0,
        out1,
        out3,
        out6,
        hidden=HIDDEN,
        reduce_blocks=REDUCE_BLOCKS,
        BLOCK_T=FINAL_BLOCKS,
        num_warps=8,
    )
    return out0, out1, add_out, out3, bf16_out, bf16_out.permute(1, 0), out6
