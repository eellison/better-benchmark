"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 layer-norm-backward-style return tuple by sharing each token row's hidden-dimension reductions across the f32 gradient output, the bf16 rounded view/transpose side outputs, and the three sibling channel reductions, whereas Inductor currently schedules the row reductions, dependent pointwise gradient, bf16 cast/view/permute fanout, and compatible `sum([0, 1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local scalar reductions with multiple channel accumulators and an observable bf16 rounding boundary; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split these layer-norm-backward tails across row tiles, reuse row scalars, and finalize the sibling channel reductions from shared partial accumulators while preserving the bf16 cast and view semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _row_stats_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_sum_ptr,
    row_dot_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    offsets = row * C + c

    x = tl.load(x_ptr + offsets, mask=c_mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weighted = x * weight

    tl.store(row_sum_ptr + row, tl.sum(tl.where(c_mask, weighted, 0.0), axis=0))
    tl.store(
        row_dot_ptr + row,
        tl.sum(tl.where(c_mask, weighted * rhs, 0.0), axis=0),
    )


@triton.jit
def _store_and_partial_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    row_sum_ptr,
    row_dot_ptr,
    grad_ptr,
    grad_bf16_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_grad_bf16_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    NORM_FACTOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    m_mask = m < M
    c_mask = c < C
    mask = m_mask[:, None] & c_mask[None, :]
    offsets = m[:, None] * C + c[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

    grad = row_scale[:, None] * (
        x * weight[None, :] * NORM_FACTOR - row_sum[:, None] - rhs * row_dot[:, None]
    )
    grad_bf16 = grad.to(tl.bfloat16)

    tl.store(grad_ptr + offsets, grad, mask=mask)
    tl.store(grad_bf16_ptr + offsets, grad_bf16, mask=mask)

    partial_offsets = tl.program_id(0) * C + c
    tl.store(
        partial_x_rhs_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * rhs, 0.0), axis=0),
        mask=c_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=c_mask,
    )
    tl.store(
        partial_grad_bf16_ptr + partial_offsets,
        tl.sum(tl.where(mask, grad_bf16.to(tl.float32), 0.0), axis=0),
        mask=c_mask,
    )


@triton.jit
def _fused_768_store_and_partial_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    grad_ptr,
    grad_bf16_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_grad_bf16_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    NORM_FACTOR: tl.constexpr,
    ROW_GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_grad_bf16 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for row_offset in tl.range(0, ROW_GROUP):
        row = group * ROW_GROUP + row_offset
        row_mask = row < M
        mask = row_mask & c_mask
        offsets = row * C + c

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row_mask, other=0.0).to(
            tl.float32
        )
        weighted = x * weight
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=0)

        grad = row_scale * (weighted * NORM_FACTOR - row_sum - rhs * row_dot)
        grad_bf16 = grad.to(tl.bfloat16)

        tl.store(grad_ptr + offsets, grad, mask=mask)
        tl.store(grad_bf16_ptr + offsets, grad_bf16, mask=mask)

        acc_x_rhs += tl.where(mask, x * rhs, 0.0)
        acc_x += tl.where(mask, x, 0.0)
        acc_grad_bf16 += tl.where(mask, grad_bf16.to(tl.float32), 0.0)

    partial_offsets = group * C + c
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
    tl.store(partial_grad_bf16_ptr + partial_offsets, acc_grad_bf16, mask=c_mask)


@triton.jit
def _finalize_channel_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_grad_bf16_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_grad_sum_ptr,
    NUM_M_TILES: tl.constexpr,
    C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C
    tiles_block = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_grad_bf16 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for tile_start in range(0, NUM_M_TILES, BLOCK_TILES):
        tiles = tile_start + tiles_block
        mask = (tiles[:, None] < NUM_M_TILES) & c_mask[None, :]
        offsets = tiles[:, None] * C + c[None, :]

        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_grad_bf16 += tl.sum(
            tl.load(partial_grad_bf16_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
    tl.store(out_x_ptr + c, acc_x, mask=c_mask)
    tl.store(out_grad_sum_ptr + c, acc_grad_bf16.to(tl.bfloat16).to(tl.float32), mask=c_mask)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# 11a5c74c: (T([32768,768], bf16), T([768], f32), T([128,256,768], f32), T([128,256,1], f32), S([128,256,768]), S([32768,768]), S([768]))
@oracle_impl(
    hardware="B200",
    point="11a5c74c",
    FUSED_768=True,
    FUSED_BLOCK_C=1024,
    ROW_BLOCK_C=1024,
    NORM_FACTOR=768,
    BLOCK_M=32,
    BLOCK_C=128,
    FINAL_BLOCK_TILES=256,
    FINAL_BLOCK_C=16,
    num_warps_stats=8,
    num_warps_store=8,
    num_warps_final=4,
)
# 96a1854b: (T([128,4096], bf16), T([4096], f32), T([1,128,4096], f32), T([1,128,1], f32), S([1,128,4096]), S([128,4096]), S([4096]))
@oracle_impl(
    hardware="B200",
    point="96a1854b",
    FUSED_768=False,
    FUSED_BLOCK_C=1,
    ROW_BLOCK_C=4096,
    NORM_FACTOR=768,
    BLOCK_M=16,
    BLOCK_C=128,
    FINAL_BLOCK_TILES=32,
    FINAL_BLOCK_C=32,
    num_warps_stats=8,
    num_warps_store=4,
    num_warps_final=4,
)
# 4676c06c: (T([4096,2048], bf16), T([2048], f32), T([32,128,2048], f32), T([32,128,1], f32), S([32,128,2048]), S([4096,2048]), S([2048]))
@oracle_impl(
    hardware="B200",
    point="4676c06c",
    FUSED_768=False,
    FUSED_BLOCK_C=1,
    ROW_BLOCK_C=2048,
    NORM_FACTOR=768,
    BLOCK_M=16,
    BLOCK_C=128,
    FINAL_BLOCK_TILES=128,
    FINAL_BLOCK_C=16,
    num_warps_stats=8,
    num_warps_store=4,
    num_warps_final=4,
)
def oracle_forward(
    inputs,
    *,
    FUSED_768: bool,
    FUSED_BLOCK_C: int,
    ROW_BLOCK_C: int,
    NORM_FACTOR: int,
    BLOCK_M: int,
    BLOCK_C: int,
    FINAL_BLOCK_TILES: int,
    FINAL_BLOCK_C: int,
    num_warps_stats: int,
    num_warps_store: int,
    num_warps_final: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs
    m = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    num_m_tiles = _cdiv(m, BLOCK_M)

    grad = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    grad_bf16 = torch.empty_strided(
        (m, c),
        (c, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (3, num_m_tiles, c),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_x_rhs = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    sum_x = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    sum_grad_bf16 = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)

    if FUSED_768:
        _fused_768_store_and_partial_kernel[(num_m_tiles,)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            grad,
            grad_bf16,
            partials[0],
            partials[1],
            partials[2],
            M=m,
            C=c,
            NORM_FACTOR=NORM_FACTOR,
            ROW_GROUP=BLOCK_M,
            BLOCK_C=FUSED_BLOCK_C,
            num_warps=num_warps_store,
            num_stages=3,
        )
    else:
        row_stats = torch.empty((2, m), device=arg0_1.device, dtype=torch.float32)
        _row_stats_kernel[(m,)](
            arg0_1,
            arg1_1,
            arg2_1,
            row_stats[0],
            row_stats[1],
            C=c,
            BLOCK_C=ROW_BLOCK_C,
            num_warps=num_warps_stats,
            num_stages=3,
        )
        _store_and_partial_kernel[(num_m_tiles, _cdiv(c, BLOCK_C))](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            row_stats[0],
            row_stats[1],
            grad,
            grad_bf16,
            partials[0],
            partials[1],
            partials[2],
            M=m,
            C=c,
            NORM_FACTOR=NORM_FACTOR,
            BLOCK_M=BLOCK_M,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps_store,
            num_stages=3,
        )
    _finalize_channel_sums_kernel[(_cdiv(c, FINAL_BLOCK_C),)](
        partials[0],
        partials[1],
        partials[2],
        sum_x_rhs,
        sum_x,
        sum_grad_bf16,
        NUM_M_TILES=num_m_tiles,
        C=c,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )

    return grad, sum_x_rhs, sum_x, grad_bf16, grad_bf16.permute(1, 0), sum_grad_bf16
