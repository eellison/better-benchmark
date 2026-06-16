"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete SigLIP class-token expand, f32-to-bf16 cast, and final bf16[128,768] view by broadcasting the single f32[1,1,768] row directly into the dense output tile, whereas Inductor lowers the captured expand/cast/view through generic pointwise materialization; Inductor cannot do this today because its pointwise codegen does not recognize singleton-token expansion as a row-broadcast cast template that hoists the source vector loads across output rows; the fix is NEW_PATTERN: add a guarded broadcast-cast materialization lowering that emits the final viewed output layout directly while preserving the exact bf16 cast and contiguous output stride."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 768


@triton.jit
def _broadcast_cast_kernel(
    x_ptr,
    out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < rows
    col_mask = col_offsets < hidden

    values = tl.load(x_ptr + col_offsets, mask=col_mask, other=0.0)
    out_offsets = row_offsets[:, None] * hidden + col_offsets[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(
        out_ptr + out_offsets,
        values[None, :].to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


@oracle_impl(hardware="B200", point="1b3ee0df", BLOCK_M=8, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    x, _shape_param_0, shape = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _broadcast_cast_kernel[(triton.cdiv(ROWS, BLOCK_M), triton.cdiv(HIDDEN, BLOCK_N))](
        x,
        out,
        rows=ROWS,
        hidden=HIDDEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
