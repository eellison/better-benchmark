"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 dropout-mask product scope by materializing the returned contiguous `[M, N]` bf16 product, returning its `[N, M]` transpose as a metadata alias, and finalizing the bf16-rounded column sum from row-tile partials, whereas Inductor lowers the bf16 casts, mask scaling, product side output, transpose alias, and dependent column reduction through generic scheduled regions; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K template that shares a materialized side-output producer with the dependent reduction while preserving bf16 rounding boundaries; the fix is COOPERATIVE_SPLIT_K: add a row-tiled product-plus-reduction template that sinks view-only layout handling and finalizes compatible column partials from the same producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _product_partial_kernel(
    values_ptr,
    keep_ptr,
    product_ptr,
    partial_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    row_block = xindex // N
    cols = xindex % N
    rows = row_block * ROW_BLOCK + tl.arange(0, RBLOCK)[None, :]

    dropout_scale = tl.full((XBLOCK, RBLOCK), 1.1111111111111112, tl.float32)
    dropout_scale_bf16 = dropout_scale.to(tl.bfloat16)
    acc = tl.zeros((XBLOCK, RBLOCK), dtype=tl.float32)

    for row_offset in tl.range(0, ROW_BLOCK, RBLOCK):
        active_rows = rows + row_offset
        mask = active_rows < M
        offsets = active_rows * N + cols

        value_bf16 = tl.load(
            values_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        keep_bf16 = tl.load(
            keep_ptr + offsets,
            mask=mask,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.float32).to(tl.bfloat16)
        keep_scaled_bf16 = (keep_bf16 * dropout_scale_bf16).to(tl.bfloat16)
        product_bf16 = (value_bf16 * keep_scaled_bf16).to(tl.bfloat16)

        tl.store(product_ptr + offsets, product_bf16, mask=mask)
        acc += tl.where(mask, product_bf16.to(tl.float32), 0.0)

    partial = tl.sum(acc, axis=1)[:, None]
    tl.store(partial_ptr + xindex, partial, mask=xindex < tl.cdiv(M, ROW_BLOCK) * N)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    out_ptr,
    N: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)[:, None]
    mask = (cols < N) & (row_blocks < NUM_ROW_BLOCKS)
    offsets = row_blocks * N + cols
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    summed = tl.sum(partials, axis=0).to(tl.bfloat16).to(tl.float32)
    out_cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(out_ptr + out_cols, summed, mask=out_cols < N)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="e6479180", XBLOCK=128, RBLOCK=16, ROW_BLOCK=128, FINAL_BLOCK_N=128, product_warps=2, final_warps=8)
@oracle_impl(hardware="B200", point="dcc62d32", XBLOCK=128, RBLOCK=16, ROW_BLOCK=128, FINAL_BLOCK_N=128, product_warps=2, final_warps=8)
@oracle_impl(hardware="B200", point="a5b2be60", XBLOCK=128, RBLOCK=16, ROW_BLOCK=128, FINAL_BLOCK_N=64, product_warps=2, final_warps=8)
@oracle_impl(hardware="B200", point="bf9ec2f8", XBLOCK=128, RBLOCK=16, ROW_BLOCK=128, FINAL_BLOCK_N=64, product_warps=2, final_warps=8)
@oracle_impl(hardware="B200", point="14e0b026", XBLOCK=128, RBLOCK=16, ROW_BLOCK=128, FINAL_BLOCK_N=32, product_warps=2, final_warps=8)
@oracle_impl(hardware="B200", point="8d0ca3a0", XBLOCK=128, RBLOCK=16, ROW_BLOCK=128, FINAL_BLOCK_N=128, product_warps=2, final_warps=8)
def oracle_forward(
    inputs,
    *,
    XBLOCK: int,
    RBLOCK: int,
    ROW_BLOCK: int,
    FINAL_BLOCK_N: int,
    product_warps: int,
    final_warps: int,
):
    values, keep, product_shape, sum_shape = inputs
    m, n = _shape_tuple(product_shape)
    (sum_n,) = _shape_tuple(sum_shape)

    product = torch.empty_strided(
        (m, n),
        (n, 1),
        device=values.device,
        dtype=torch.bfloat16,
    )
    num_row_blocks = triton.cdiv(m, ROW_BLOCK)
    partials = torch.empty_strided(
        (num_row_blocks, n),
        (n, 1),
        device=values.device,
        dtype=torch.float32,
    )
    summed = torch.empty_strided(
        (sum_n,),
        (1,),
        device=values.device,
        dtype=torch.float32,
    )

    _product_partial_kernel[(triton.cdiv(num_row_blocks * n, XBLOCK),)](
        values,
        keep,
        product,
        partials,
        N=n,
        M=m,
        ROW_BLOCK=ROW_BLOCK,
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
        num_warps=product_warps,
        num_stages=1,
    )
    _finalize_sum_kernel[(triton.cdiv(n, FINAL_BLOCK_N),)](
        partials,
        summed,
        N=n,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=final_warps,
        num_stages=1,
    )

    return product, product.permute(1, 0), summed
