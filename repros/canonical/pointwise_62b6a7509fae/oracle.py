"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BEiT relative-position-bias gather by writing the indexed `[732,12]` table directly into the returned head-major padded bf16 backing tensor `[1,12,197,200]`, zero-filling the right pad columns, and returning the `[128,12,197,197]` zero-stride expanded slice alias from that same storage; Inductor lowers the gather, permute/clone, bf16 cast, pad, slice, and expand as separate generic layout concerns; Inductor cannot do this today because it lacks a gather-to-padded-layout lowering that recognizes pad-then-slice as a destination-layout request on top of an indexed table lookup; the fix is NEW_PATTERN: add a relative-position-bias gather template that writes the final padded backing storage directly and preserves the expanded view metadata."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 197
HEADS = 12
PADDED_SEQ = 200
BATCH = 128
PAD_COLS = PADDED_SEQ - SEQ
VISIBLE_ELEMENTS = HEADS * SEQ * SEQ
PAD_ELEMENTS = HEADS * SEQ * PAD_COLS
BASE_SHAPE = (1, HEADS, SEQ, PADDED_SEQ)
BASE_STRIDE = (HEADS * SEQ * PADDED_SEQ, SEQ * PADDED_SEQ, PADDED_SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)
EXPAND_STRIDE = (0, SEQ * PADDED_SEQ, PADDED_SEQ, 1)


@triton.jit
def _relative_bias_linear_kernel(
    index_ptr,
    table_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    SEQ_C: tl.constexpr,
    HEADS_C: tl.constexpr,
    PADDED_SEQ_C: tl.constexpr,
    PAD_COLS_C: tl.constexpr,
    PAD_ELEMENTS_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    hw = SEQ_C * SEQ_C
    head = offsets // hw
    pos = offsets - head * hw
    row = pos // SEQ_C
    col = pos - row * SEQ_C
    visible_mask = mask & (offsets < (HEADS_C * hw))
    indices = tl.load(index_ptr + row * SEQ_C + col, mask=visible_mask, other=0)
    values = tl.load(
        table_ptr + indices * HEADS_C + head,
        mask=visible_mask,
        other=0.0,
    )
    out_offsets = head * (SEQ_C * PADDED_SEQ_C) + row * PADDED_SEQ_C + col
    tl.store(
        out_ptr + out_offsets,
        values,
        mask=visible_mask,
    )

    pad_mask = mask & (offsets < PAD_ELEMENTS_C)
    pad_col = SEQ_C + (offsets % PAD_COLS_C)
    row_head = offsets // PAD_COLS_C
    pad_row = row_head % SEQ_C
    pad_head = row_head // SEQ_C
    pad_out = pad_head * (SEQ_C * PADDED_SEQ_C) + pad_row * PADDED_SEQ_C + pad_col
    tl.store(out_ptr + pad_out, tl.zeros((BLOCK,), tl.float32), mask=pad_mask)


@oracle_impl(hardware="B200", point="0b512413", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    index, table, *_shape_params = inputs
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=table.device,
        dtype=torch.bfloat16,
    )
    total = max(VISIBLE_ELEMENTS, PAD_ELEMENTS)
    _relative_bias_linear_kernel[(triton.cdiv(total, BLOCK),)](
        index,
        table,
        base,
        TOTAL=total,
        SEQ_C=SEQ,
        HEADS_C=HEADS,
        PADDED_SEQ_C=PADDED_SEQ,
        PAD_COLS_C=PAD_COLS,
        PAD_ELEMENTS_C=PAD_ELEMENTS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    expanded = torch.as_strided(base, EXPAND_SHAPE, EXPAND_STRIDE)
    return base, expanded
