"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete SigLIP bf16 layernorm-backward fragment by sharing the hidden-dimension row reductions with the returned bf16 strided view, the batch-reduced fp32 side output, and sibling channel reductions, whereas Inductor schedules the row reductions, dependent pointwise epilogue, dtype-rounding boundary, side-output reduction, and channel reductions as separate generic kernels; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, required materialized views, and multiple dependent reductions in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a split-row/token reduction template that fuses layernorm-backward epilogues with compatible side-output and channel reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 256
CHANNELS = 768
ROWS = BATCH * TOKENS
ROW_BLOCK = 16
ROW_C_BLOCK = 1024
EPI_C_BLOCK = 16
FINAL_C_BLOCK = 16
FINAL_T_BLOCK = 256
NUM_TOKENS = TOKENS
VIEW_SHAPE = (BATCH, CHANNELS, 16, 16)
VIEW_STRIDE = (TOKENS * CHANNELS, 1, 16 * CHANNELS, CHANNELS)
SUM5_SHAPE = (1, TOKENS, CHANNELS)


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
    norm_src_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    row_sum_ptr,
    row_dot_ptr,
    rows_total: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < rows_total
    col_mask = cols < channels
    mask = row_mask[:, None] & col_mask[None, :]
    token = rows - (rows // tokens) * tokens
    offsets = rows[:, None] * channels + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    src = tl.load(norm_src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(
        bias_ptr + token[:, None] * channels + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    weighted = _f32_mul(x, gamma[None, :])
    norm = _f32_mul(_f32_sub(_f32_add(src, bias), mean[:, None]), invstd[:, None])
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, _f32_mul(weighted, norm), 0.0), axis=1)

    tl.store(row_sum_ptr + rows, row_sum, mask=row_mask)
    tl.store(row_dot_ptr + rows, row_dot, mask=row_mask)


@triton.jit
def _token_epilogue_kernel(
    x_ptr,
    gamma_ptr,
    norm_src_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    residual_ptr,
    row_sum_ptr,
    row_dot_ptr,
    sum5_ptr,
    view_out_ptr,
    partial_ptr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    c_tile = tl.program_id(1)
    batches = tl.arange(0, BLOCK_B)
    cols = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = batches * tokens + token
    col_mask = cols < channels
    mask = col_mask[None, :]
    offsets = rows[:, None] * channels + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    src = tl.load(norm_src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(
        bias_ptr + token * channels + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(mean_ptr + rows).to(tl.float32)
    invstd = tl.load(invstd_ptr + rows).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + rows).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + rows).to(tl.float32)

    weighted = _f32_mul(x, gamma[None, :])
    norm = _f32_mul(_f32_sub(_f32_add(src, bias[None, :]), mean[:, None]), invstd[:, None])
    centered = _f32_sub(_f32_mul(weighted, 768.0), row_sum[:, None])
    centered = _f32_sub(centered, _f32_mul(norm, row_dot[:, None]))
    scale = _f32_div(invstd[:, None], 768.0)
    grad = _f32_mul(scale, centered)
    add_value = _f32_add(residual, grad)
    add_bf16_f32 = _round_bf16_to_f32(add_value)

    tl.store(view_out_ptr + offsets, add_bf16_f32, mask=mask)
    tl.store(
        sum5_ptr + token * channels + cols,
        tl.sum(tl.where(mask, add_value, 0.0), axis=0),
        mask=col_mask,
    )

    partial_offset = token * channels + cols
    plane = tokens * channels
    tl.store(
        partial_ptr + partial_offset,
        tl.sum(tl.where(mask, _f32_mul(x, norm), 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partial_ptr + plane + partial_offset,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partial_ptr + 2 * plane + partial_offset,
        tl.sum(tl.where(mask, add_bf16_f32, 0.0), axis=0),
        mask=col_mask,
    )


@triton.jit
def _finalize_kernel(
    partial_ptr,
    out0_ptr,
    out1_ptr,
    out4_ptr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_tile = tl.program_id(0)
    token_offsets = tl.arange(0, BLOCK_T)
    cols = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (token_offsets[:, None] < tokens) & (cols[None, :] < channels)
    offsets = token_offsets[:, None] * channels + cols[None, :]
    plane = tokens * channels

    out0 = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=0)
    out1 = tl.sum(tl.load(partial_ptr + plane + offsets, mask=mask, other=0.0), axis=0)
    out4 = tl.sum(
        tl.load(partial_ptr + 2 * plane + offsets, mask=mask, other=0.0),
        axis=0,
    )
    col_mask = cols < channels
    tl.store(out0_ptr + cols, out0, mask=col_mask)
    tl.store(out1_ptr + cols, out1, mask=col_mask)
    tl.store(out4_ptr + cols, _round_bf16_to_f32(out4), mask=col_mask)


@oracle_impl(hardware="B200", point="9362191f")
def oracle_forward(inputs):
    x, gamma, norm_src, bias, mean, invstd, residual, *_shape_params = inputs
    device = x.device

    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    sum5 = torch.empty(SUM5_SHAPE, device=device, dtype=torch.float32)
    view_out = torch.empty_strided(
        VIEW_SHAPE,
        VIEW_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty((3, NUM_TOKENS, CHANNELS), device=device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out4 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _row_reductions_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        x,
        gamma,
        norm_src,
        bias,
        mean,
        invstd,
        row_sum,
        row_dot,
        rows_total=ROWS,
        tokens=TOKENS,
        channels=CHANNELS,
        BLOCK_R=ROW_BLOCK,
        BLOCK_C=ROW_C_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _token_epilogue_kernel[(TOKENS, triton.cdiv(CHANNELS, EPI_C_BLOCK))](
        x,
        gamma,
        norm_src,
        bias,
        mean,
        invstd,
        residual,
        row_sum,
        row_dot,
        sum5,
        view_out,
        partial,
        tokens=TOKENS,
        channels=CHANNELS,
        BLOCK_B=BATCH,
        BLOCK_C=EPI_C_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _finalize_kernel[(triton.cdiv(CHANNELS, FINAL_C_BLOCK),)](
        partial,
        out0,
        out1,
        out4,
        tokens=TOKENS,
        channels=CHANNELS,
        BLOCK_T=FINAL_T_BLOCK,
        BLOCK_C=FINAL_C_BLOCK,
        num_warps=8,
        num_stages=4,
    )

    return out0, out1, sum5, view_out, out4
