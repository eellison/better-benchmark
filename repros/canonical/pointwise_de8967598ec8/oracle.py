"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-2 bf16 segment-aware causal attention-mask scope by materializing the twelve distinct `[8,1,1024,1024]` 0/-inf outputs directly from the per-batch segment ids and causal relation, whereas Inductor lowers the iota/advanced-index/equality/causal-mask/where chain through generic pointwise scheduling and repeats the same bf16 mask stores for each returned tensor; Inductor cannot do this today because it has no guarded segment-causal-mask materialization pattern that recognizes the paired segment gathers and repeated sibling where outputs as one attention-mask idiom while preserving distinct output storages; the fix is NEW_PATTERN: add a segment-aware causal-mask lowering that emits the compact repeated bf16 materializations directly with exact Inf behavior and full output scope."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
N_ELEMENTS = BATCH * SEQ * SEQ
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@triton.jit
def _segment_causal_mask_12_kernel(
    segments,
    out0,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    out7,
    out8,
    out9,
    out10,
    out11,
    S: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < TOTAL
    col = offsets % S
    row = (offsets // S) % S
    batch = offsets // (S * S)
    base = segments + batch * S

    row_segment = tl.load(base + row, mask=valid, other=0)
    col_segment = tl.load(base + col, mask=valid, other=1)
    keep = (col <= row) & (row_segment == col_segment)
    value = tl.where(keep, 0.0, -float("inf"))

    tl.store(out0 + offsets, value, mask=valid)
    tl.store(out1 + offsets, value, mask=valid)
    tl.store(out2 + offsets, value, mask=valid)
    tl.store(out3 + offsets, value, mask=valid)
    tl.store(out4 + offsets, value, mask=valid)
    tl.store(out5 + offsets, value, mask=valid)
    tl.store(out6 + offsets, value, mask=valid)
    tl.store(out7 + offsets, value, mask=valid)
    tl.store(out8 + offsets, value, mask=valid)
    tl.store(out9 + offsets, value, mask=valid)
    tl.store(out10 + offsets, value, mask=valid)
    tl.store(out11 + offsets, value, mask=valid)


@oracle_impl(hardware="B200", point="0105f520", BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    segments, _shape0 = inputs
    outputs = [
        torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=segments.device,
            dtype=torch.bfloat16,
        )
        for _ in range(12)
    ]
    grid = (triton.cdiv(N_ELEMENTS, BLOCK),)
    _segment_causal_mask_12_kernel[grid](
        segments,
        *outputs,
        S=SEQ,
        TOTAL=N_ELEMENTS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return tuple(outputs)
