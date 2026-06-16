"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 right-column `constant_pad_nd` scope by directly materializing the fresh contiguous `[256,197952]` output, copying the non-contiguous `[256,197951]` input view whose row stride already matches the padded width and zero-filling only the final column of each row, whereas Inductor's isolated pad lowering already has the same mandatory read/write memory envelope for this repro; Inductor cannot materially avoid the full input read, full output write, allocation, and one launch because there is no producer or consumer fusion left to remove; the fix is BANDWIDTH_BOUND: record this as an at-floor pad-copy case unless broader pad-copy memory codegen, allocation, or launch-overhead work moves both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 256
IN_COLS = 197951
OUT_COLS = 197952
OUT_NUMEL = ROWS * OUT_COLS


@triton.jit
def _right_pad_strided_view_kernel(
    input_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    col = offsets % OUT_WIDTH
    copy_mask = mask & (col < (OUT_WIDTH - 1))
    values = tl.load(input_ptr + offsets, mask=copy_mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


# (T([256,197951], bf16, stride=(197952,1)))
@oracle_impl(hardware="B200", point="3a1fd470", BLOCK=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (x,) = inputs
    out = torch.empty_strided(
        (ROWS, OUT_COLS),
        (OUT_COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _right_pad_strided_view_kernel[(triton.cdiv(OUT_NUMEL, BLOCK),)](
        x,
        out,
        TOTAL=OUT_NUMEL,
        OUT_WIDTH=OUT_COLS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
