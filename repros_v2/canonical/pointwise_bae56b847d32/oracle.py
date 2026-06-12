"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BEiT relative-position-bias gather by writing indexed bf16 table values directly into the returned contiguous `[1,12,197,197]` unsqueezed clone layout, whereas Inductor lowers the gather, view, permute, clone, and unsqueeze as generic advanced-indexing and layout materialization work; Inductor cannot do this today because it lacks a destination-layout-aware relative-position gather template that maps the indexed `[732,12]` table directly into head-major output storage; the fix is NEW_PATTERN: add a guarded relative-position-bias gather lowering that combines the indexed table lookup with the final clone layout store while preserving dtype, output scope, and strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 197
HEADS = 12
ELEMENTS = SEQ * SEQ
OUT_SHAPE = (1, HEADS, SEQ, SEQ)
OUT_STRIDE = (HEADS * ELEMENTS, ELEMENTS, SEQ, 1)


@triton.jit
def _relative_bias_linear_kernel(
    index_ptr,
    table_ptr,
    out_ptr,
    SEQ_C: tl.constexpr,
    HEADS_C: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    hw = SEQ_C * SEQ_C
    head = offsets // hw
    spatial = offsets - head * hw
    indices = tl.load(index_ptr + spatial, mask=mask, other=0)
    values = tl.load(table_ptr + indices * HEADS_C + head, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="86e210b7", BLOCK=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    index, table, _shape_param_0 = inputs
    del _shape_param_0

    out = torch.empty(OUT_SHAPE, device=table.device, dtype=table.dtype)
    _relative_bias_linear_kernel[(triton.cdiv(HEADS * ELEMENTS, BLOCK),)](
        index,
        table,
        out,
        SEQ_C=SEQ,
        HEADS_C=HEADS,
        TOTAL=HEADS * ELEMENTS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
