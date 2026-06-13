"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 BEiT class-token LayerNorm-backward redistribution fragment by sharing the 128 row-local hidden reductions, writing the observable f32 `[128,197,768]` slice_scatter output, writing the bf16 projected `[25216,768]` output plus its transpose alias, and finalizing all four returned hidden-column reductions from common token tiles, whereas Inductor schedules the row reductions, expanded slice_scatter materialization, two dependent projection reductions, bf16 cast/view fanout, final rounded sum, and alias-only transpose return as separate generic regions; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that keeps row-local LayerNorm-backward summaries, explicit expansion/slice_scatter layout, dtype-rounding boundaries, and compatible column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible LayerNorm-backward column reductions across token tiles while sinking the f32/bf16 side-output stores and metadata-only transpose return into the same producer/finalizer schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
PATCH_TOKENS = 196
HIDDEN = 768
TOKEN_BLOCK = 16
CHANNEL_BLOCK = 64
ROW_CHANNEL_BLOCK = 1024
FINAL_CHANNEL_BLOCK = 8
TOKEN_TILES = triton.cdiv(TOKENS, TOKEN_BLOCK)
PARTIAL_TILES = BATCH * TOKEN_TILES
FINAL_TOKEN_BLOCK = 2048


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
def _row_summary_kernel(
    x_ptr,
    gamma_ptr,
    norm_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    HIDDEN_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < HIDDEN_
    offsets = batch * HIDDEN_ + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weighted = _f32_mul(x, gamma)

    tl.store(row_sum_ptr + batch, tl.sum(tl.where(mask, weighted, 0.0), axis=0))
    tl.store(
        row_dot_ptr + batch,
        tl.sum(tl.where(mask, _f32_mul(weighted, norm), 0.0), axis=0),
    )
    tl.store(partial_x_norm_ptr + offsets, _f32_mul(x, norm), mask=mask)
    tl.store(partial_x_ptr + offsets, x, mask=mask)


@triton.jit
def _materialize_kernel(
    x_ptr,
    gamma_ptr,
    norm_ptr,
    inv_scale_ptr,
    proj_weight_ptr,
    other_ptr,
    row_sum_ptr,
    row_dot_ptr,
    scatter_out_ptr,
    bf16_out_ptr,
    partial_other_ptr,
    partial_bf16_sum_ptr,
    TOKEN_TILES_: tl.constexpr,
    TOKENS_: tl.constexpr,
    PATCH_TOKENS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    tile_t = tl.program_id(1)
    tile_c = tl.program_id(2)
    tokens = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
    token_mask = tokens < TOKENS_
    col_mask = cols < HIDDEN_
    mask = token_mask[:, None] & col_mask[None, :]

    x_offsets = batch * HIDDEN_ + cols
    x = tl.load(x_ptr + x_offsets, mask=col_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + x_offsets, mask=col_mask, other=0.0).to(tl.float32)
    inv_scale = tl.load(inv_scale_ptr + batch).to(tl.float32)
    proj_weight = tl.load(proj_weight_ptr + cols, mask=col_mask, other=0.0).to(
        tl.float32
    )
    row_sum = tl.load(row_sum_ptr + batch).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + batch).to(tl.float32)

    weighted = _f32_mul(x, gamma)
    term = _f32_sub(_f32_mul(weighted, 768.0), row_sum)
    term = _f32_sub(term, _f32_mul(norm, row_dot))
    grad = _f32_mul(inv_scale, term)
    patch_value = _f32_div(grad, 196.0)

    token_is_patch = tokens > 0
    scatter_value = tl.where(token_is_patch[:, None], patch_value[None, :], 0.0)
    out_offsets = (batch * TOKENS_ + tokens[:, None]) * HIDDEN_ + cols[None, :]
    tl.store(scatter_out_ptr + out_offsets, scatter_value, mask=mask)

    projected = _f32_mul(scatter_value, proj_weight[None, :])
    projected_bf16 = projected.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(bf16_out_ptr + out_offsets, projected_bf16, mask=mask)

    other = tl.load(other_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
    partial_offsets = (batch * TOKEN_TILES_ + tile_t) * HIDDEN_ + cols
    tl.store(
        partial_other_ptr + partial_offsets,
        tl.sum(tl.where(mask, _f32_mul(scatter_value, other), 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partial_bf16_sum_ptr + partial_offsets,
        tl.sum(tl.where(mask, projected_bf16.to(tl.float32), 0.0), axis=0),
        mask=col_mask,
    )


@triton.jit
def _finalize_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    partial_other_ptr,
    partial_bf16_sum_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_other_ptr,
    out_bf16_sum_ptr,
    BATCH_: tl.constexpr,
    PARTIAL_TILES_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_

    batch_offsets = tl.arange(0, BLOCK_B)
    batch_mask = (batch_offsets[:, None] < BATCH_) & col_mask[None, :]
    batch_ptrs = batch_offsets[:, None] * HIDDEN_ + cols[None, :]
    out0 = tl.sum(
        tl.load(partial_x_norm_ptr + batch_ptrs, mask=batch_mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    out1 = tl.sum(
        tl.load(partial_x_ptr + batch_ptrs, mask=batch_mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )

    tile_offsets = tl.arange(0, BLOCK_TILES)
    tile_mask = (tile_offsets[:, None] < PARTIAL_TILES_) & col_mask[None, :]
    tile_ptrs = tile_offsets[:, None] * HIDDEN_ + cols[None, :]
    out3 = tl.sum(
        tl.load(partial_other_ptr + tile_ptrs, mask=tile_mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    out6 = tl.sum(
        tl.load(partial_bf16_sum_ptr + tile_ptrs, mask=tile_mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )

    tl.store(out_x_norm_ptr + cols, out0, mask=col_mask)
    tl.store(out_x_ptr + cols, out1, mask=col_mask)
    tl.store(out_other_ptr + cols, out3, mask=col_mask)
    tl.store(out_bf16_sum_ptr + cols, out6.to(tl.bfloat16).to(tl.float32), mask=col_mask)


# 6da1d727: BEiT train class-token LN-backward redistribution, f32 slice_scatter, bf16 projected view.
@oracle_impl(
    hardware="B200",
    point="6da1d727",
    token_warps=4,
    final_warps=8,
)
def oracle_forward(inputs, *, token_warps: int, final_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        shape_param_1,
        shape_param_2,
        _shape_param_3,
        shape_param_4,
        _shape_param_5,
    ) = inputs
    del _shape_param_0, _shape_param_3, _shape_param_5

    device = arg0_1.device
    row_sum = torch.empty((BATCH,), device=device, dtype=torch.float32)
    row_dot = torch.empty((BATCH,), device=device, dtype=torch.float32)
    partial_x_norm = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial_x = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial_other = torch.empty((PARTIAL_TILES, HIDDEN), device=device, dtype=torch.float32)
    partial_bf16_sum = torch.empty(
        (PARTIAL_TILES, HIDDEN),
        device=device,
        dtype=torch.float32,
    )
    out_x_norm = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    scatter_out = torch.empty_strided(
        tuple(shape_param_1),
        (TOKENS * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    out_other = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    bf16_out = torch.empty_strided(
        tuple(shape_param_4),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_bf16_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    _row_summary_kernel[(BATCH,)](
        arg0_1,
        arg1_1,
        arg2_1,
        row_sum,
        row_dot,
        partial_x_norm,
        partial_x,
        HIDDEN_=HIDDEN,
        BLOCK_C=ROW_CHANNEL_BLOCK,
        num_warps=8,
    )
    _materialize_kernel[(BATCH, TOKEN_TILES, triton.cdiv(HIDDEN, CHANNEL_BLOCK))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        row_sum,
        row_dot,
        scatter_out,
        bf16_out,
        partial_other,
        partial_bf16_sum,
        TOKEN_TILES_=TOKEN_TILES,
        TOKENS_=TOKENS,
        PATCH_TOKENS_=PATCH_TOKENS,
        HIDDEN_=HIDDEN,
        BLOCK_T=TOKEN_BLOCK,
        BLOCK_C=CHANNEL_BLOCK,
        num_warps=token_warps,
    )
    _finalize_kernel[(triton.cdiv(HIDDEN, FINAL_CHANNEL_BLOCK),)](
        partial_x_norm,
        partial_x,
        partial_other,
        partial_bf16_sum,
        out_x_norm,
        out_x,
        out_other,
        out_bf16_sum,
        BATCH_=BATCH,
        PARTIAL_TILES_=PARTIAL_TILES,
        HIDDEN_=HIDDEN,
        BLOCK_B=128,
        BLOCK_TILES=FINAL_TOKEN_BLOCK,
        BLOCK_C=FINAL_CHANNEL_BLOCK,
        num_warps=final_warps,
    )

    return (
        out_x_norm,
        out_x,
        scatter_out,
        out_other,
        bf16_out,
        bf16_out.permute(1, 0),
        out_bf16_sum,
    )
