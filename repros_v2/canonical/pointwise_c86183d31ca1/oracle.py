"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete f32 identity multiply, token-id validity mask, fresh zero-filled vocabulary output, and duplicate-preserving `_unsafe_masked_index_put_accumulate` as a direct row-wise Triton scatter-add for every captured shape point, whereas Inductor lowers the same mask/full/scatter graph through generic pointwise and indexed-update codegen; Inductor cannot do this today because scheduler/codegen does not canonicalize a broadcast masked vocabulary `index_put` accumulate into a single scatter-reduce plan with the mask folded into the indexed update; the fix is SCATTER_REDUCE: add a guarded lowering for masked row scatter-adds that emits the dense zero-fill and duplicate-safe indexed accumulation directly."""

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
def _masked_vocab_scatter_kernel(
    values_ptr,
    index_ptr,
    out_ptr,
    values_stride0: tl.constexpr,
    values_stride1: tl.constexpr,
    values_stride2: tl.constexpr,
    index_stride0: tl.constexpr,
    index_stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    seq_len: tl.constexpr,
    total_sources: tl.constexpr,
    out_rows: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    source = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden

    batch = source // seq_len
    token = source - batch * seq_len
    vocab_row = tl.load(index_ptr + batch * index_stride0 + token * index_stride1).to(tl.int64)
    active = (
        (source < total_sources)
        & (vocab_row >= 0)
        & (vocab_row < out_rows)
        & (vocab_row != 1)
    )

    values = tl.load(
        values_ptr
        + batch * values_stride0
        + token * values_stride1
        + cols * values_stride2,
        mask=active & col_mask,
        other=0.0,
    ).to(tl.float32) * 1.0

    tl.atomic_add(
        out_ptr + vocab_row * out_stride0 + cols * out_stride1,
        values,
        sem="relaxed",
        mask=active & col_mask,
    )


# 19f6778a: (T([8,1024,1024], f32), T([8,1024], i64), S([50265,1024]))
@oracle_impl(hardware="B200", point="19f6778a", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# 47b53512: (T([32,128,2560], f32), T([32,128], i64), S([8008,2560]))
@oracle_impl(hardware="B200", point="47b53512", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# c51d1d47: (T([16,128,2560], f32), T([16,128], i64), S([8008,2560]))
@oracle_impl(hardware="B200", point="c51d1d47", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# f18349c3: (T([64,128,1024], f32), T([64,128], i64), S([128112,1024]))
@oracle_impl(hardware="B200", point="f18349c3", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# f507fb35: (T([16,1024,768], f32), T([16,1024], i64), S([50005,768]))
@oracle_impl(hardware="B200", point="f507fb35", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# 79ee2993: (T([64,256,1024], f32), T([64,256], i64), S([50265,1024]))
@oracle_impl(hardware="B200", point="79ee2993", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
# e91334e5: (T([32,128,1024], f32), T([32,128], i64), S([256008,1024]))
@oracle_impl(hardware="B200", point="e91334e5", ZERO_BLOCK=1024, BLOCK_H=1024, scatter_warps=4)
def oracle_forward(inputs, *, ZERO_BLOCK, BLOCK_H, scatter_warps):
    values, indices, out_shape = inputs
    out_rows = int(out_shape[0])
    hidden = int(out_shape[1])
    batch = int(indices.shape[0])
    seq_len = int(indices.shape[1])
    total_sources = batch * seq_len
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
    _masked_vocab_scatter_kernel[(total_sources, triton.cdiv(hidden, BLOCK_H))](
        values,
        indices,
        out,
        values_stride0=int(values.stride(0)),
        values_stride1=int(values.stride(1)),
        values_stride2=int(values.stride(2)),
        index_stride0=int(indices.stride(0)),
        index_stride1=int(indices.stride(1)),
        out_stride0=int(out.stride(0)),
        out_stride1=int(out.stride(1)),
        seq_len=seq_len,
        total_sources=total_sources,
        out_rows=out_rows,
        hidden=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=scatter_warps,
    )
    return out
