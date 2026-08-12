"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MoE routing postprocess by preserving ATen's exact `topk` and `sort` ordering boundaries, then fusing the sorted-position mask, int32 cast, and bf16 row materialization in Triton, whereas Inductor lowers the decomposed topk/view/sort/ge/unsqueeze/index/where/cast graph as separate generic library and pointwise/gather stages; Inductor cannot do this today because its scheduler/codegen has no routed-topk post-sort gather template that recognizes the sorted-token contract and emits one materialization epilogue; the fix is NEW_PATTERN: add a guarded MoE routing postprocess lowering around the exact topk/sort boundary with a fused sorted-row gather and metadata epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 1000
EXPERTS = 32
TOPK = 4
TOTAL_TOPK = ROWS * TOPK
HIDDEN = 2880


@triton.jit
def _routing_metadata_kernel(
    sorted_ids_ptr,
    mask_ptr,
    cast_ptr,
    n_elements: tl.constexpr,
    valid_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    ids = tl.load(sorted_ids_ptr + offsets, mask=mask, other=0)
    invalid = ids >= valid_experts
    tl.store(mask_ptr + offsets, invalid, mask=mask)
    tl.store(cast_ptr + offsets, ids.to(tl.int32), mask=mask)


@triton.jit
def _sorted_row_gather_kernel(
    rows_ptr,
    sorted_ids_ptr,
    sort_positions_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    topk: tl.constexpr,
    valid_experts: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = cols < n_cols

    expert = tl.load(sorted_ids_ptr + row)
    position = tl.load(sort_positions_ptr + row)
    source_row = position // topk
    valid = expert < valid_experts

    values = tl.load(
        rows_ptr + source_row * n_cols + cols,
        mask=col_mask & valid,
        other=0.0,
    )
    values = tl.where(valid, values, 0.0)
    tl.store(out_ptr + row * n_cols + cols, values, mask=col_mask)


# a9d89d2e: (T([1000,32], bf16), T([1000,2880], bf16))
@oracle_impl(
    hardware="B200",
    point="a9d89d2e",
    BLOCK_META=1024,
    BLOCK_COLS=1024,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_META, BLOCK_COLS, num_warps):
    arg0_1, arg1_1 = inputs

    topk_values, topk_indices = torch.topk(arg0_1, TOPK)
    flat_indices = topk_indices.view(-1)
    sorted_ids, sort_positions = torch.sort(flat_indices)

    unsqueeze = torch.empty((TOTAL_TOPK, 1), device=arg0_1.device, dtype=torch.bool)
    where = torch.empty((TOTAL_TOPK, HIDDEN), device=arg1_1.device, dtype=torch.bfloat16)
    cast = torch.empty((TOTAL_TOPK,), device=arg0_1.device, dtype=torch.int32)

    _routing_metadata_kernel[(triton.cdiv(TOTAL_TOPK, BLOCK_META),)](
        sorted_ids,
        unsqueeze,
        cast,
        n_elements=TOTAL_TOPK,
        valid_experts=EXPERTS,
        BLOCK=BLOCK_META,
        num_warps=num_warps,
    )
    _sorted_row_gather_kernel[(TOTAL_TOPK, triton.cdiv(HIDDEN, BLOCK_COLS))](
        arg1_1,
        sorted_ids,
        sort_positions,
        where,
        n_cols=HIDDEN,
        topk=TOPK,
        valid_experts=EXPERTS,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=num_warps,
    )
    return topk_values, sorted_ids, sort_positions, unsqueeze, where, cast
