"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete T5 bf16 RMSNorm/dropout-backward scope from Repro.forward, including the bf16 input cast, both bool-mask scales, the [0, 1] weight-gradient sum, the row-local hidden reduction, the returned f32 add tensor, the bf16 dropout-gradient view, and its transposed view, whereas Inductor schedules the shared producer, sibling reductions, bf16 epilogue, and view-return envelope as generic split kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that coordinates a row-local reduction, a full dense side-output store, a sibling column accumulator, and the required bf16 rounding boundary; the fix is COOPERATIVE_SPLIT_K: add a guarded RMSNorm/dropout-backward template that splits the row domain, combines column partials, and fuses the dependent dense epilogue while preserving visible output strides and casts."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 8192
H = 512
SCALE_X = 0.04419417382415922
MASK_SCALE = 1.1111111111111112
INV_H = 0.001953125


@triton.jit
def _row_epilogue_and_partials_kernel(
    x_ptr,
    mask0_ptr,
    weight_ptr,
    activ_ptr,
    row_scale_ptr,
    mask1_ptr,
    partial_col_ptr,
    add_out_ptr,
    bf16_out_ptr,
    BLOCK_M: tl.constexpr,
    H_: tl.constexpr,
    SCALE_X_: tl.constexpr,
    MASK_SCALE_: tl.constexpr,
    INV_H_: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, H_)
    offsets = rows[:, None] * H_ + cols[None, :]

    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    mask0 = tl.load(mask0_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    activ = tl.load(activ_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, eviction_policy="evict_last").to(tl.float32)
    mask1 = tl.load(mask1_ptr + offsets, eviction_policy="evict_first").to(tl.float32)

    scaled_x = x * SCALE_X_
    scaled_mask0 = mask0 * MASK_SCALE_
    producer = scaled_x * scaled_mask0

    mul4 = activ * row_scale[:, None]
    col_term = producer * mul4
    partial = tl.sum(col_term, axis=0)
    tl.store(partial_col_ptr + tl.program_id(0) * H_ + cols, partial)

    weighted = producer * weight[None, :]
    row_term = weighted * activ
    row_sum = tl.sum(row_term, axis=1)

    first = weighted * row_scale[:, None]
    row_scale_sq = row_scale * row_scale
    row_scale_cu = row_scale_sq * row_scale
    norm_term = row_sum * -0.5
    norm_term = norm_term * row_scale_cu
    norm_term = norm_term * INV_H_
    activ_twice = activ * 2.0
    second = norm_term[:, None] * activ_twice
    add_out = first + second
    tl.store(add_out_ptr + offsets, add_out)

    add_bf16 = add_out.to(tl.bfloat16).to(tl.float32)
    mask1_scaled = (mask1 * MASK_SCALE_).to(tl.bfloat16).to(tl.float32)
    bf16_out = (add_bf16 * mask1_scaled).to(tl.bfloat16)
    tl.store(bf16_out_ptr + offsets, bf16_out)


@triton.jit
def _finalize_column_sum_kernel(
    partial_col_ptr,
    vector_out_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    H_: tl.constexpr,
):
    col = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_R)
    mask = row_blocks < NUM_ROW_BLOCKS
    vals = tl.load(
        partial_col_ptr + row_blocks * H_ + col,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    tl.store(vector_out_ptr + col, tl.sum(vals, axis=0))


# a4350e46: T5 train bf16 RMSNorm/dropout backward, rows=8192, hidden=512.
@oracle_impl(hardware="B200", point="a4350e46", BLOCK_M=2, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    x, mask0, weight, activ, row_scale, mask1, shape_3d, shape_vec, shape_3d_b, shape_2d = inputs
    del shape_3d, shape_vec, shape_3d_b, shape_2d

    num_row_blocks = triton.cdiv(M, BLOCK_M)
    partial_col = torch.empty((num_row_blocks, H), device=x.device, dtype=torch.float32)
    vector_out = torch.empty((H,), device=x.device, dtype=torch.float32)
    add_out = torch.empty_strided((8, 1024, H), (1024 * H, H, 1), device=x.device, dtype=torch.float32)
    bf16_out = torch.empty_strided((M, H), (H, 1), device=x.device, dtype=torch.bfloat16)

    _row_epilogue_and_partials_kernel[(num_row_blocks,)](
        x,
        mask0,
        weight,
        activ,
        row_scale,
        mask1,
        partial_col,
        add_out,
        bf16_out,
        BLOCK_M=BLOCK_M,
        H_=H,
        SCALE_X_=SCALE_X,
        MASK_SCALE_=MASK_SCALE,
        INV_H_=INV_H,
        num_warps=num_warps,
        num_stages=3,
    )

    _finalize_column_sum_kernel[(H,)](
        partial_col,
        vector_out,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_R=triton.next_power_of_2(num_row_blocks),
        H_=H,
        num_warps=8,
        num_stages=3,
    )

    return vector_out, add_out, bf16_out, bf16_out.permute(1, 0)
