"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete validity-mask construction, fresh zero-filled f32 output, and duplicate-preserving `_unsafe_masked_index_put_accumulate` as a direct row-wise Triton scatter-add for every captured rank-1 and rank-2 token point, whereas Inductor lowers the mask/full/scatter graph through generic pointwise and indexed-update codegen; Inductor cannot do this today because scheduler/codegen does not canonicalize a broadcast masked row `index_put` accumulate into a single scatter-reduce plan with the validity mask folded into the indexed update; the fix is SCATTER_REDUCE: add a guarded lowering for masked row scatter-adds that emits the dense zero-fill and duplicate-safe indexed accumulation directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _zero_kernel(
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + offsets,
        tl.zeros((BLOCK,), tl.float32),
        mask=offsets < n_elements,
    )


@triton.jit
def _masked_row_scatter_kernel(
    index_ptr,
    values_ptr,
    out_ptr,
    total_sources: tl.constexpr,
    out_rows: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    source = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden

    row = tl.load(index_ptr + source).to(tl.int64)
    active = (source < total_sources) & (row >= 0) & (row < out_rows) & (row != -1)
    values = tl.load(
        values_ptr + source * hidden + cols,
        mask=active & col_mask,
        other=0.0,
    ).to(tl.float32)

    tl.atomic_add(
        out_ptr + row * hidden + cols,
        values,
        sem="relaxed",
        mask=active & col_mask,
    )


# 1bba723b: (T([1,1024], i64), T([1,1024,1024], f32), S([1026,1024]))
@oracle_impl(hardware="B200", point="1bba723b", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# 154a9c83: (T([128], i64), T([128,2560], f32), S([128,2560]))
@oracle_impl(hardware="B200", point="154a9c83", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=8)
# b92ebe46: (T([4,2048], i64), T([4,2048,768], f32), S([2050,768]))
@oracle_impl(hardware="B200", point="b92ebe46", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# e0a3bc30: (T([1,1024], i64), T([1,1024,768], f32), S([1026,768]))
@oracle_impl(hardware="B200", point="e0a3bc30", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# 79ee2993: (T([64,256], i64), T([64,256,1024], f32), S([514,1024]))
@oracle_impl(hardware="B200", point="79ee2993", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
def oracle_forward(inputs, *, ZERO_BLOCK, BLOCK_H, scatter_warps):
    indices, values, out_shape = inputs
    out_rows = int(out_shape[0])
    hidden = int(out_shape[1])
    total_sources = int(indices.numel())
    n_elements = out_rows * hidden

    out = torch.empty_strided(
        (out_rows, hidden),
        (hidden, 1),
        device=values.device,
        dtype=torch.float32,
    )

    _zero_kernel[(triton.cdiv(n_elements, ZERO_BLOCK),)](
        out,
        n_elements=n_elements,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _masked_row_scatter_kernel[(total_sources, triton.cdiv(hidden, BLOCK_H))](
        indices,
        values,
        out,
        total_sources=total_sources,
        out_rows=out_rows,
        hidden=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=scatter_warps,
    )
    return out
