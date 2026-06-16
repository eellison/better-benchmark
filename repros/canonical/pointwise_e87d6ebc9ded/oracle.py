"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full attention-head layout materialization by writing the fresh contiguous view/permute/clone/view result directly from the contiguous bf16 `[B*H, S, D]` input into the returned `[B*S, H*D]` tensor with one Triton copy kernel, whereas Inductor currently emits a generic layout materialization for the captured reshape/permute/clone/reshape chain; Inductor cannot do this today because the scheduler/codegen path does not recognize this common attention head-major to sequence-major clone as a specialized layout-copy pattern with simplified affine indexing; the fix is NEW_PATTERN: add a guarded attention head-layout materialization template for `view(B,H,S,D).permute(0,2,1,3).contiguous().view(B*S,H*D)`."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_sequence_layout_kernel(
    input_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    mask = (rows < ROWS) & (cols < HIDDEN)

    batch = rows // SEQ
    seq = rows - batch * SEQ
    head = cols // HEAD_DIM
    dim = cols - head * HEAD_DIM

    input_offsets = (
        batch * (HEADS * SEQ * HEAD_DIM)
        + head * (SEQ * HEAD_DIM)
        + seq * HEAD_DIM
        + dim
    )
    output_offsets = rows * HIDDEN + cols

    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="fb089404")
@oracle_impl(hardware="B200", point="226fbbfa")
@oracle_impl(hardware="B200", point="c6b0f684")
@oracle_impl(hardware="B200", point="07e248d7")
@oracle_impl(hardware="B200", point="576ca76e")
@oracle_impl(hardware="B200", point="471d82af")
@oracle_impl(hardware="B200", point="c6cb1dd8")
@oracle_impl(hardware="B200", point="e6f344ac")
@oracle_impl(hardware="B200", point="2cdbce9d")
@oracle_impl(hardware="B200", point="c23ba4e7")
@oracle_impl(hardware="B200", point="6c3c2efc")
@oracle_impl(hardware="B200", point="a3cab238")
@oracle_impl(hardware="B200", point="14c0be85")
@oracle_impl(hardware="B200", point="d528e08b")
@oracle_impl(hardware="B200", point="1dcf8636")
def oracle_forward(inputs):
    arg0, shape0, _shape1, shape2 = inputs
    heads = int(shape0[1])
    seq = int(arg0.shape[1])
    head_dim = int(arg0.shape[2])
    hidden = heads * head_dim
    rows = int(shape2[0])

    output = torch.empty_strided(
        (rows, int(shape2[1])),
        (int(shape2[1]), 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    block_cols = 128 if hidden <= 128 else 512 if hidden <= 768 else 1024
    block_rows = 1 if hidden >= 2560 or rows <= 1000 else 2 if hidden >= 1024 else 4
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(hidden, block_cols))
    _head_sequence_layout_kernel[grid](
        arg0,
        output,
        ROWS=rows,
        SEQ=seq,
        HEADS=heads,
        HEAD_DIM=head_dim,
        HIDDEN=hidden,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )
    return output
