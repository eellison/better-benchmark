"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete OPT bf16 layer-norm/dropout backward fragment by row-tiling the `[8192, 768]` producer, recomputing the normalized residual branch, writing the returned f32 `[4, 2048, 768]` gradient side output, writing the returned bf16 masked `[8192, 768]` output and its `[768, 8192]` transpose view alias, and cooperatively accumulating the three returned `[768]` sums from the same row tiles; Inductor currently emits separate generic kernels for the residual normalization producer, row-local layer-norm backward reductions, the f32 side output, the bf16 mask/scale side output, and the sibling column reductions; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output reduction template that keeps row-local reductions, dtype-boundary-preserving side-output stores, alias-preserving transpose returns, and sibling column reductions in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add row-tiled multi-output reduction scheduling that fuses the layer-norm backward epilogue with compatible column partials and layout side outputs while preserving the captured bf16/f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 8192
D = 768
DROP_SCALE = 1.1111111111111112
INV_D = 0.0013020833333333333


@triton.jit
def _row_tile_kernel(
    x_ptr,
    gamma_ptr,
    residual_ptr,
    mask_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    grad_in_ptr,
    grad_out_ptr,
    masked_out_ptr,
    partials_ptr,
    NUM_TILES: tl.constexpr,
    D_: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D_
    gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    acc_x_norm = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for row_off in tl.range(0, ROWS_PER_TILE):
        row = tile * ROWS_PER_TILE + row_off
        offsets = row * D_ + d
        x = tl.load(x_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
        keep = tl.load(mask_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row).to(tl.float32)
        invstd = tl.load(invstd_ptr + row).to(tl.float32)
        grad_in = tl.load(grad_in_ptr + offsets, mask=d_mask, other=0.0).to(tl.float32)

        residual_masked = (keep * residual).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        )
        residual_scaled = (residual_masked.to(tl.float32) * 1.1111111111111112).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        )
        norm = (bias + residual_scaled.to(tl.float32) - mean) * invstd
        weighted = x * gamma
        row_sum = tl.sum(tl.where(d_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(d_mask, weighted * norm, 0.0), axis=0)
        grad = grad_in + (invstd * 0.0013020833333333333) * (weighted * (D_ + 0.0) - row_sum - norm * row_dot)
        tl.store(grad_out_ptr + offsets, grad, mask=d_mask)

        grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne")
        keep_scaled = (keep * 1.1111111111111112).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        )
        masked = (grad_bf16.to(tl.float32) * keep_scaled.to(tl.float32)).to(
            tl.bfloat16,
            fp_downcast_rounding="rtne",
        )
        tl.store(masked_out_ptr + offsets, masked, mask=d_mask)

        acc_x_norm += tl.where(d_mask, x * norm, 0.0)
        acc_x += tl.where(d_mask, x, 0.0)
        acc_masked += tl.where(d_mask, masked.to(tl.float32), 0.0)

    partial_offsets = tile * D_ + d
    partial_stride = NUM_TILES * D_
    tl.store(partials_ptr + partial_offsets, acc_x_norm, mask=d_mask)
    tl.store(partials_ptr + partial_stride + partial_offsets, acc_x, mask=d_mask)
    tl.store(partials_ptr + partial_stride * 2 + partial_offsets, acc_masked, mask=d_mask)


@triton.jit
def _finalize_kernel(
    partials_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_masked_ptr,
    NUM_TILES: tl.constexpr,
    D_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    tiles = tl.arange(0, BLOCK_TILES)[:, None]
    d_b = d[None, :]
    mask = (tiles < NUM_TILES) & (d_b < D_)
    offsets = tiles * D_ + d_b
    partial_stride = NUM_TILES * D_

    x_norm = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0), axis=0)
    x_sum = tl.sum(
        tl.load(partials_ptr + partial_stride + offsets, mask=mask, other=0.0),
        axis=0,
    )
    masked_sum = tl.sum(
        tl.load(partials_ptr + partial_stride * 2 + offsets, mask=mask, other=0.0),
        axis=0,
    )

    d_mask = d < D_
    tl.store(out_x_norm_ptr + d, x_norm, mask=d_mask)
    tl.store(out_x_ptr + d, x_sum, mask=d_mask)
    rounded_masked_sum = masked_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(
        tl.float32
    )
    tl.store(out_masked_ptr + d, rounded_masked_sum, mask=d_mask)


@oracle_impl(
    hardware="B200",
    point="81e4a7b8",
    ROWS_PER_TILE=16,
    BLOCK_D=1024,
    FINAL_BLOCK_TILES=512,
    FINAL_BLOCK_D=16,
    row_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_TILE: int,
    BLOCK_D: int,
    FINAL_BLOCK_TILES: int,
    FINAL_BLOCK_D: int,
    row_warps: int,
    final_warps: int,
):
    (
        x,
        gamma,
        residual,
        mask,
        bias,
        mean,
        invstd,
        grad_in,
        *_shape_args,
    ) = inputs

    num_tiles = triton.cdiv(M, ROWS_PER_TILE)
    grad_out = torch.empty_strided(
        (4, 2048, D),
        (2048 * D, D, 1),
        device=x.device,
        dtype=torch.float32,
    )
    masked_out = torch.empty_strided((M, D), (D, 1), device=x.device, dtype=torch.bfloat16)
    partials = torch.empty((3, num_tiles, D), device=x.device, dtype=torch.float32)
    out_x_norm = torch.empty((D,), device=x.device, dtype=torch.float32)
    out_x = torch.empty((D,), device=x.device, dtype=torch.float32)
    out_masked = torch.empty((D,), device=x.device, dtype=torch.float32)

    _row_tile_kernel[(num_tiles,)](
        x,
        gamma,
        residual,
        mask,
        bias,
        mean,
        invstd,
        grad_in,
        grad_out,
        masked_out,
        partials,
        NUM_TILES=num_tiles,
        D_=D,
        ROWS_PER_TILE=ROWS_PER_TILE,
        BLOCK_D=BLOCK_D,
        num_warps=row_warps,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partials,
        out_x_norm,
        out_x,
        out_masked,
        NUM_TILES=num_tiles,
        D_=D,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=final_warps,
        num_stages=4,
    )
    return out_x_norm, out_x, grad_out, masked_out, masked_out.permute(1, 0), out_masked
