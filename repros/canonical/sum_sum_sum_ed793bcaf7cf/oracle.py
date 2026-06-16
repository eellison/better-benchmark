"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16/fp32 DistillGPT2 layernorm-backward/dropout return tuple by row-tiling the `[M, 768]` producer, preserving the bf16 input conversion, f32 add input, row-local hidden reductions, f32 gradient output, two sibling `[768]` column reductions, bf16 dropout-product rounding, returned `[M, 768]` dropout view, and bf16-rounded final dropout sum from one producer/finalizer pair, whereas Inductor lowers the row reductions, dense f32 epilogue, bf16 mask product, returned side output, and column reductions through separate generic schedules that reread materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local reductions feeding dependent column reductions with live returned tensor stores and bf16 dtype boundaries; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layernorm-backward reductions across row tiles, keep row-local summaries in the producer, and finalize multiple column accumulators while sinking returned side-output stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _row_partial_kernel(
    x_bf16_ptr,
    addend_ptr,
    weight_ptr,
    xhat_ptr,
    scale_ptr,
    keep_ptr,
    grad_out_ptr,
    drop_out_ptr,
    partial_x_xhat_ptr,
    partial_x_ptr,
    partial_drop_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_m = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    dropout_scale = tl.full((BLOCK_D,), 1.1111111111111112, tl.float32)
    dropout_scale_bf16 = dropout_scale.to(tl.bfloat16)

    acc_x_xhat = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_drop = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for row_inner in tl.static_range(0, BLOCK_M):
        row = tile_m * BLOCK_M + row_inner
        row_mask = row < M
        mask = row_mask & col_mask
        offsets = row * D + cols

        x_base = tl.load(x_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        addend = tl.load(addend_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        x = addend + x_base
        weighted = x * weight
        row_sum = tl.sum(tl.where(col_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(col_mask, weighted * xhat, 0.0), axis=0)
        grad = scale * (weighted * D - row_sum - xhat * row_dot)
        tl.store(grad_out_ptr + offsets, grad, mask=mask)

        dropped_bf16 = (grad * keep * dropout_scale).to(tl.bfloat16)
        tl.store(drop_out_ptr + offsets, dropped_bf16, mask=mask)

        grad_bf16 = grad.to(tl.bfloat16)
        keep_bf16 = keep.to(tl.bfloat16)
        keep_scaled_bf16 = (keep_bf16 * dropout_scale_bf16).to(tl.bfloat16)
        dropped_sum_bf16 = (grad_bf16 * keep_scaled_bf16).to(tl.bfloat16)

        acc_x_xhat += tl.where(mask, x * xhat, 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_drop += tl.where(mask, dropped_sum_bf16.to(tl.float32), 0.0)

    partial_offsets = tile_m * D + cols
    tl.store(partial_x_xhat_ptr + partial_offsets, acc_x_xhat, mask=col_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)
    tl.store(partial_drop_ptr + partial_offsets, acc_drop, mask=col_mask)


@triton.jit
def _finalize_kernel(
    partial_x_xhat_ptr,
    partial_x_ptr,
    partial_drop_ptr,
    out_x_xhat_ptr,
    out_x_ptr,
    out_drop_ptr,
    NUM_TILES: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    tiles = tl.arange(0, BLOCK_TILES)[:, None]
    mask = (tiles < NUM_TILES) & (cols < D)
    offsets = tiles * D + cols

    x_xhat = tl.load(partial_x_xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dropped = tl.load(partial_drop_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out_cols = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    out_mask = out_cols < D
    tl.store(out_x_xhat_ptr + out_cols, tl.sum(x_xhat, axis=0), mask=out_mask)
    tl.store(out_x_ptr + out_cols, tl.sum(x, axis=0), mask=out_mask)
    rounded_drop = tl.sum(dropped, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_drop_ptr + out_cols, rounded_drop, mask=out_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 9846b7f2: (T([16384,768], bf16), T([32,512,768], f32), T([768], f32), ...)
@oracle_impl(hardware="B200", point="9846b7f2", BLOCK_M=16, BLOCK_D=1024, FINAL_BLOCK_D=8, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_D: int,
    FINAL_BLOCK_D: int,
    num_warps: int,
):
    x_bf16, addend, weight, xhat, scale, keep, shape0, shape1, shape2 = inputs
    b, s, d = _shape_tuple(shape0)
    m = b * s

    grad_out = torch.empty_strided(
        (b, s, d),
        (s * d, d, 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    drop_out = torch.empty_strided(
        _shape_tuple(shape1),
        (d, 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    out_x_xhat = torch.empty_strided(_shape_tuple(shape2), (1,), device=x_bf16.device, dtype=torch.float32)
    out_x = torch.empty_strided(_shape_tuple(shape2), (1,), device=x_bf16.device, dtype=torch.float32)
    out_drop = torch.empty_strided(_shape_tuple(shape2), (1,), device=x_bf16.device, dtype=torch.float32)

    num_tiles = triton.cdiv(m, BLOCK_M)
    partial_x_xhat = torch.empty_strided((num_tiles, d), (d, 1), device=x_bf16.device, dtype=torch.float32)
    partial_x = torch.empty_strided((num_tiles, d), (d, 1), device=x_bf16.device, dtype=torch.float32)
    partial_drop = torch.empty_strided((num_tiles, d), (d, 1), device=x_bf16.device, dtype=torch.float32)

    _row_partial_kernel[(num_tiles,)](
        x_bf16,
        addend,
        weight,
        xhat,
        scale,
        keep,
        grad_out,
        drop_out,
        partial_x_xhat,
        partial_x,
        partial_drop,
        M=m,
        D=d,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    block_tiles = 1 << (num_tiles - 1).bit_length()
    _finalize_kernel[(triton.cdiv(d, FINAL_BLOCK_D),)](
        partial_x_xhat,
        partial_x,
        partial_drop,
        out_x_xhat,
        out_x,
        out_drop,
        NUM_TILES=num_tiles,
        D=d,
        BLOCK_TILES=block_tiles,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
    )

    return grad_out, out_x_xhat, out_x, drop_out, out_drop
