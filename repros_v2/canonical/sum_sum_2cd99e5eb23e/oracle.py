"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full static RMSNorm backward scope with the same mixed-order schedule as the best prior art: each row-block program computes the row-wise dot needed for the bf16 input-gradient epilogue while accumulating partial column sums for the returned f32 weight gradient, whereas Inductor already uses a comparable fused row epilogue plus partial column-reduction plan and mainly pays the unavoidable full bf16 matrix read/write and f32 partial traffic; Inductor cannot materially improve this local pattern without reducing that required memory traffic or launch overhead; the fix is BANDWIDTH_BOUND: keep the fused row-block layout, preserve the exact f32/bf16 cast boundaries and operation order, and collapse the partial column sums once."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 1_152_000
N = 512
ROWS_PER_BLOCK = 64
XBLOCK = 8
PIPELINE_STAGES = 8
INV_N = 0.001953125


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
def _rmsnorm_bwd_row_kernel(
    scale_ptr,
    weight_ptr,
    rstd_ptr,
    x_ptr,
    out_ptr,
    partial_ptr,
    M_: tl.constexpr,
    N_: tl.constexpr,
    ROWS_PER_BLOCK_: tl.constexpr,
    XBLOCK_: tl.constexpr,
    PIPELINE_STAGES_: tl.constexpr,
    INV_N_: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, N_)
    row_index = pid * ROWS_PER_BLOCK_ + tl.arange(0, XBLOCK_)

    scale = tl.load(scale_ptr).to(tl.float32)
    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    scaled_weight = _f32_mul(scale, weight)
    accum = tl.full([N_], 0.0, tl.float32)

    for _ in tl.range(0, ROWS_PER_BLOCK_, XBLOCK_, num_stages=PIPELINE_STAGES_):
        row_mask = row_index < M_
        offsets = row_index[:, None] * N_ + cols[None, :]
        x = tl.load(
            x_ptr + offsets,
            mask=row_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        rstd = tl.load(
            rstd_ptr + row_index,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)[:, None]

        weighted_x = _f32_mul(scaled_weight[None, :], x)
        row_sum = tl.sum(tl.where(row_mask[:, None], weighted_x, 0.0), axis=1)[:, None]
        row_scale = _f32_mul(_f32_mul(row_sum, -0.5), _f32_mul(_f32_mul(rstd, rstd), rstd))
        row_scale = _f32_mul(row_scale, INV_N_)
        first_term = _f32_mul(scaled_weight[None, :], rstd)
        second_term = _f32_mul(row_scale, _f32_mul(x, 2.0))
        out = _f32_add(first_term, second_term)

        col_terms = _f32_mul(_f32_mul(x, rstd), scale)
        accum += tl.sum(tl.where(row_mask[:, None], col_terms, 0.0), axis=0)
        tl.store(out_ptr + offsets, out, mask=row_mask[:, None])
        row_index += XBLOCK_

    tl.store(partial_ptr + pid * N_ + cols, accum)


@triton.jit
def _finalize_cols_kernel(
    partial_ptr,
    out_ptr,
    N_: tl.constexpr,
    NUM_BLOCKS_: tl.constexpr,
    BLOCKS_P2: tl.constexpr,
):
    col = tl.program_id(0)
    blocks = tl.arange(0, BLOCKS_P2)
    mask = blocks < NUM_BLOCKS_
    values = tl.load(partial_ptr + blocks * N_ + col, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr + col, total)


@oracle_impl(
    hardware="B200",
    point="cbc6f48e",
    rows_per_block=64,
    xblock=8,
    pipeline_stages=8,
    row_warps=2,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    rows_per_block: int,
    xblock: int,
    pipeline_stages: int,
    row_warps: int,
    final_warps: int,
):
    scale, weight, rstd, x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2
    num_row_blocks = triton.cdiv(M, rows_per_block)

    out = torch.empty_strided((M, N), (N, 1), device=x.device, dtype=torch.bfloat16)
    partials = torch.empty((num_row_blocks, N), device=x.device, dtype=torch.float32)
    grad_weight = torch.empty((N,), device=x.device, dtype=torch.float32)

    _rmsnorm_bwd_row_kernel[(num_row_blocks,)](
        scale,
        weight,
        rstd,
        x,
        out,
        partials,
        M_=M,
        N_=N,
        ROWS_PER_BLOCK_=rows_per_block,
        XBLOCK_=xblock,
        PIPELINE_STAGES_=pipeline_stages,
        INV_N_=INV_N,
        num_warps=row_warps,
        num_stages=1,
    )
    _finalize_cols_kernel[(N,)](
        partials,
        grad_weight,
        N_=N,
        NUM_BLOCKS_=num_row_blocks,
        BLOCKS_P2=_ceil_pow2(num_row_blocks),
        num_warps=final_warps,
        num_stages=3,
    )
    return out, grad_weight
