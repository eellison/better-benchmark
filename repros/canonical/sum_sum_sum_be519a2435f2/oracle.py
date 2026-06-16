"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete FNet layer-norm-backward/dropout return tuple by reducing each hidden-size-768 token row once, writing both returned full-scope f32 tensors (`mul_4` and the dropout-scaled `[16384,768]` view), returning the required `[768,16384]` transpose alias, and cooperatively accumulating the two pre-mask `[768]` column reductions plus the post-mask `[768]` column reduction from the same row-tiled producer, whereas Inductor currently schedules the row reductions, dropout-mask pointwise epilogue, view/permute side outputs, and sibling `sum([0,1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, dependent full-tensor stores, and multiple sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across token tiles, finalize their partial accumulators, and fuse both full-tensor stores plus the transpose-view return."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 16384
C = 768
DROP_SCALE = 1.1111111111111112


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
def _row_group_store_and_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    drop_mask_ptr,
    grad_out_ptr,
    masked_out_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_ptr,
    M_: tl.constexpr,
    C_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    ROW_GROUP: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C_
    weight = tl.load(weight_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for group_offset in tl.range(0, ROW_GROUP, XBLOCK):
        rows = row_group * ROW_GROUP + group_offset + tl.arange(0, XBLOCK)
        row_mask = rows < M_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = rows[:, None] * C_ + cols[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        drop_mask = tl.load(drop_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)

        weighted = _f32_mul(x, weight[None, :])
        weighted_scaled = _f32_mul(weighted, 768.0)
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        weighted_rhs = _f32_mul(weighted, rhs)
        row_dot = tl.sum(tl.where(mask, weighted_rhs, 0.0), axis=1)
        centered = _f32_sub(weighted_scaled, row_sum[:, None])
        centered = _f32_sub(centered, _f32_mul(rhs, row_dot[:, None]))
        grad = _f32_mul(row_scale[:, None], centered)

        mask_scaled = _f32_mul(drop_mask, DROP_SCALE_)
        masked = _f32_mul(grad, mask_scaled)
        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(masked_out_ptr + offsets, masked, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, _f32_mul(x, rhs), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_masked += tl.sum(tl.where(mask, masked, 0.0), axis=0)

    partial_offsets = row_group * C_ + cols
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
    tl.store(partial_masked_ptr + partial_offsets, acc_masked, mask=c_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_masked_ptr,
    NUM_M_TILES: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = cols < C_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_masked = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for tile_start in tl.range(0, NUM_M_TILES, BLOCK_TILES):
        tiles = tile_start + tile_offsets
        mask = (tiles[:, None] < NUM_M_TILES) & c_mask[None, :]
        offsets = tiles[:, None] * C_ + cols[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_masked += tl.sum(
            tl.load(partial_masked_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + cols, acc_x_rhs, mask=c_mask)
    tl.store(out_x_ptr + cols, acc_x, mask=c_mask)
    tl.store(out_masked_ptr + cols, acc_masked, mask=c_mask)


# (T([16384,768], f32), T([768], f32), T([32,512,768], f32), T([32,512,1], f32), T([32,512,768], b8), S([32,512,768]), S([16384,768]), S([768]))
@oracle_impl(hardware="B200", point="0d0b1d4e", ROW_GROUP=22, XBLOCK=1, BLOCK_C=1024, FINAL_BLOCK_C=16, FINAL_BLOCK_TILES=512, row_warps=4, final_warps=8)
def oracle_forward(
    inputs,
    *,
    ROW_GROUP: int,
    XBLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    FINAL_BLOCK_TILES: int,
    row_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    device = arg0_1.device
    num_m_tiles = triton.cdiv(M, ROW_GROUP)

    grad_out = torch.empty_strided(
        (32, 512, C),
        (512 * C, C, 1),
        device=device,
        dtype=torch.float32,
    )
    masked_out = torch.empty_strided((M, C), (C, 1), device=device, dtype=torch.float32)
    partial_x_rhs = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_masked = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)

    _row_group_store_and_reduce_kernel[(num_m_tiles,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        grad_out,
        masked_out,
        partial_x_rhs,
        partial_x,
        partial_masked,
        M_=M,
        C_=C,
        DROP_SCALE_=DROP_SCALE,
        ROW_GROUP=ROW_GROUP,
        XBLOCK=XBLOCK,
        BLOCK_C=BLOCK_C,
        num_warps=row_warps,
        num_stages=3,
    )

    out_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_masked = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_masked,
        out_x_rhs,
        out_x,
        out_masked,
        NUM_M_TILES=num_m_tiles,
        C_=C,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )

    return grad_out, out_x_rhs, out_x, masked_out, masked_out.permute(1, 0), out_masked
