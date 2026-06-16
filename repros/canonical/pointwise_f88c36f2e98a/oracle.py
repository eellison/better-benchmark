"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 bottom-row `constant_pad_nd` scope by directly materializing the fresh contiguous `[197952,512]` output, copying the contiguous `[197951,512]` input prefix and zero-filling the single appended row, whereas Inductor's isolated pad lowering already has the same mandatory memory envelope for this repro; Inductor cannot materially avoid the full input read, full output write, allocation, and one launch because there is no producer or consumer fusion left to remove; the fix is BANDWIDTH_BOUND: record this as an at-floor pad-copy case unless broader pad-copy memory codegen, allocation, or launch-overhead work moves both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


IN_ROWS = 197951
COLS = 512
OUT_ROWS = 197952
IN_NUMEL = IN_ROWS * COLS
OUT_NUMEL = OUT_ROWS * COLS


@triton.jit
def _bottom_pad_contiguous_kernel(
    input_ptr,
    output_ptr,
    IN_TOTAL: tl.constexpr,
    OUT_TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    out_mask = offsets < OUT_TOTAL
    copy_mask = offsets < IN_TOTAL
    values = tl.load(input_ptr + offsets, mask=copy_mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=out_mask)


# (T([197951,512], bf16))
@oracle_impl(hardware="B200", point="580b2d03", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (x,) = inputs
    out = torch.empty_strided(
        (OUT_ROWS, COLS),
        (COLS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bottom_pad_contiguous_kernel[(triton.cdiv(OUT_NUMEL, BLOCK),)](
        x,
        out,
        IN_TOTAL=IN_NUMEL,
        OUT_TOTAL=OUT_NUMEL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
