"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-OSS indexed SwiGLU-style bf16 pointwise scope, including the visible clamped index output and the contiguous `[4000,2880]` gated output, in one row/column Triton schedule that avoids Inductor's separate index-clamp launch and flat div/mod indexing in the main pointwise kernel while preserving the captured bf16 rounding boundaries and final strides; Inductor cannot do this today because its pointwise scheduler does not form a single multi-output gather-plus-gated-activation template for this paired clamp/index/slice pattern; the fix is SCHEDULER_FUSION: teach pointwise codegen to fuse small index side outputs into the consumer gather kernel and specialize split-last-dimension indexing."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=8, num_stages=4),
    ],
    key=["N_ROWS", "N_COLS"],
)
@triton.jit
def _indexed_swiglu_kernel(
    index_in,
    table,
    values,
    index_out,
    out,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    INPUT_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < N_ROWS
    col_mask = col_offsets < N_COLS
    mask = row_mask[:, None] & col_mask[None, :]

    raw_index = tl.load(index_in + row_offsets, mask=row_mask, other=0)
    clamped = triton_helpers.minimum(raw_index, tl.full([BLOCK_M], 31, tl.int64))
    if tl.program_id(1) == 0:
        tl.store(index_out + row_offsets, clamped, mask=row_mask)

    wrapped_index = tl.where(clamped < 0, clamped + 32, clamped)
    even_cols = col_offsets * 2
    odd_cols = even_cols + 1
    value_base = row_offsets[:, None] * INPUT_COLS
    table_base = wrapped_index[:, None] * INPUT_COLS

    odd_value = tl.load(
        values + value_base + odd_cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    odd_table = tl.load(
        table + table_base + odd_cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    odd_sum = (odd_value + odd_table).to(tl.bfloat16).to(tl.float32)
    odd_clamped = triton_helpers.minimum(
        triton_helpers.maximum(odd_sum, -7.0),
        7.0,
    ).to(tl.bfloat16).to(tl.float32)
    odd_term = (odd_clamped + 1.0).to(tl.bfloat16).to(tl.float32)

    even_value = tl.load(
        values + value_base + even_cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    even_table = tl.load(
        table + table_base + even_cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    even_sum = (even_value + even_table).to(tl.bfloat16).to(tl.float32)
    even_clamped = triton_helpers.minimum(even_sum, 7.0).to(tl.bfloat16).to(tl.float32)
    gate = (even_clamped * 1.702).to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    swish = (even_clamped * sigmoid).to(tl.bfloat16).to(tl.float32)
    result = (odd_term * swish).to(tl.bfloat16)

    out_offsets = row_offsets[:, None] * N_COLS + col_offsets[None, :]
    tl.store(out + out_offsets, result, mask=mask)


@oracle_impl(hardware="B200", point="55c3c977")
def oracle_forward(inputs):
    index_in, table, values = inputs
    rows = int(values.shape[0])
    input_cols = int(values.shape[1])
    out_cols = input_cols // 2
    index_out = torch.empty_strided(
        (rows,),
        (1,),
        device=index_in.device,
        dtype=torch.int64,
    )
    out = torch.empty_strided(
        (rows, out_cols),
        (out_cols, 1),
        device=values.device,
        dtype=torch.bfloat16,
    )
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_M"]),
        triton.cdiv(out_cols, meta["BLOCK_N"]),
    )
    _indexed_swiglu_kernel[grid](
        index_in,
        table,
        values,
        index_out,
        out,
        N_ROWS=rows,
        N_COLS=out_cols,
        INPUT_COLS=input_cols,
    )
    return index_out, out
