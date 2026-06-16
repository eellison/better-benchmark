"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16-to-f32 three-input residual add directly in the returned `[8,1024,768]` layout, folding the three flat `[8192,768] -> [1024,8,768]` views and the `[1,0,2]` permute into output-space indexing while preserving the f32 add order and final contiguous f32 output; Inductor lowers the captured view/cast/add/permute/add scope through a generic flattened pointwise schedule that does not select a guarded sequence/batch/hidden layout template for this Longformer row transform; the fix is NEW_PATTERN: add a fixed-hidden Longformer sequence-major to batch-major pointwise template that promotes bf16 inputs once, sinks metadata-only views into addressing, and writes the contiguous residual-add output directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _longformer_three_bf16_residual_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    residual_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    batch = rows // 1024
    seq = rows - batch * 1024
    source_rows = seq * 8 + batch

    mask = (rows < 8192) & (cols < 768)
    source_offsets = source_rows * 768 + cols
    output_offsets = rows * 768 + cols

    value0 = tl.load(arg0_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    value1 = tl.load(arg1_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    value2 = tl.load(arg2_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + output_offsets, mask=mask, other=0.0)

    sum01 = value0 + value1
    sum012 = sum01 + value2
    out = residual + sum012
    tl.store(out_ptr + output_offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="b940b015", BLOCK_M=4, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    output = torch.empty_strided(
        (8, 1024, 768),
        (786432, 768, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(8192, BLOCK_M), triton.cdiv(768, BLOCK_N))
    _longformer_three_bf16_residual_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return output
