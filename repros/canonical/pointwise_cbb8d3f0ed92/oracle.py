"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Blenderbot full-true masked row index-put accumulation by zero-initializing the dense f32[128,2560] result and scattering each source row directly into duplicate destination rows, whereas Inductor lowers the full/full/_unsafe_masked_index_put_accumulate graph as generic fill plus indexed update work; Inductor cannot do this today because scheduler/codegen does not recognize a full-true masked row index-put accumulate as a structured scatter-reduce with the mask folded away; the fix is SCATTER_REDUCE: add an index-put-accumulate lowering that folds full-true masks and emits a direct duplicate-preserving row scatter/reduce schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
COLS = 2560


@triton.jit
def _row_index_put_scatter_kernel(
    index_ptr,
    values_ptr,
    out_ptr,
    index_stride0: tl.constexpr,
    values_stride0: tl.constexpr,
    values_stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    rows: tl.constexpr,
    cols_total: tl.constexpr,
    block_cols: tl.constexpr,
):
    src_row = tl.program_id(0)
    cols = tl.program_id(1) * block_cols + tl.arange(0, block_cols)
    col_mask = cols < cols_total
    raw_index = tl.load(index_ptr + src_row * index_stride0).to(tl.int64)
    wrapped_index = tl.where(raw_index < 0, raw_index + rows, raw_index)
    row_mask = (wrapped_index >= 0) & (wrapped_index < rows)

    values = tl.load(
        values_ptr + src_row * values_stride0 + cols * values_stride1,
        mask=col_mask & row_mask,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        out_ptr + wrapped_index * out_stride0 + cols * out_stride1,
        values,
        sem="relaxed",
        mask=col_mask & row_mask,
    )


@oracle_impl(hardware="B200", point="154a9c83", block_cols=256, num_warps=4)
def oracle_forward(inputs, *, block_cols: int, num_warps: int):
    index, values, _shape0, _shape1 = inputs
    out = torch.empty_strided((ROWS, COLS), (COLS, 1), device=values.device, dtype=torch.float32)
    out.zero_()
    _row_index_put_scatter_kernel[(ROWS, triton.cdiv(COLS, block_cols))](
        index,
        values,
        out,
        index_stride0=index.stride(0),
        values_stride0=values.stride(0),
        values_stride1=values.stride(1),
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        rows=ROWS,
        cols_total=COLS,
        block_cols=block_cols,
        num_warps=num_warps,
    )
    return out
