"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvBert attention-cache layout concat by directly materializing the final contiguous `[16384,768]` tensor from the permuted logical `[32,512,6,64]` view of `arg0` and the reshaped `[32,512,6,64]` view of `arg1`; Inductor lowers the permute/clone/view/cat/view chain as generic layout materialization around a cat instead of recognizing the fixed two-half head concat as one output-layout copy; the fix is SCHEDULER_FUSION: add a guarded attention head-layout concat lowering that writes both 384-column halves of the flattened output in a single tiled pass."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_concat_kernel(
    permuted_ptr,
    reshaped_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = cols < 768

    first_half = cols < 384
    first = tl.load(permuted_ptr + rows * 384 + cols, mask=mask & first_half, other=0.0)
    second = tl.load(reshaped_ptr + rows * 384 + (cols - 384), mask=mask & ~first_half, other=0.0)
    values = tl.where(first_half, first, second)
    tl.store(out_ptr + rows * 768 + cols, values, mask=mask)


@oracle_impl(hardware="B200", point="add2068b", BLOCK_M=2, BLOCK_N=1024, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    n = int(_shape_param_3[0])
    width = int(_shape_param_3[1])
    out = torch.empty_strided((n, width), (width, 1), device=arg0_1.device, dtype=arg0_1.dtype)
    _head_concat_kernel[(triton.cdiv(width, BLOCK_N), triton.cdiv(n, BLOCK_M))](
        arg0_1,
        arg1_1,
        out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
