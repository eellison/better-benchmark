"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GPT-Neo attention row-projection scope in one Triton row-reduction kernel, including the [512,128,128] to [32,16,128,128] view, bf16-to-fp32 promotion, fp32 product and last-dimension sum, prims.fma-equivalent projection, explicit bf16 rounding, sliced broadcast bool mask, scalar bf16 fill, and final [512,128,128] contiguous output, whereas Inductor lowers the decomposed view/convert/mul/sum/fma/convert/slice/where/view graph through its generic fused reduction scheduler; Inductor cannot do this today because the scheduler does not recognize this fixed-width row projection with a required bf16 rounding boundary and repeated mask epilogue as one shape-specialized full-scope template; the fix is SCHEDULER_FUSION: add a row-projection reduction template that fuses the producer, fma epilogue, dtype boundary, mask fill, and final view store."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _row_projection_mask_kernel(
    arg0_ptr,
    arg1_ptr,
    mask_ptr,
    fill_ptr,
    out_ptr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    offsets = rows * BLOCK_N + cols
    queries = rows % BLOCK_N

    x = tl.load(arg0_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    w = tl.load(arg1_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    product = x * w
    row_sum = tl.sum(product, axis=1)[:, None]
    projected = tl.fma(-w, row_sum, product)

    keep = tl.load(mask_ptr + queries * 2048 + cols, eviction_policy="evict_last").to(tl.int1)
    fill = tl.load(fill_ptr)
    out = tl.where(keep, projected, fill)
    tl.store(out_ptr + offsets, out)


# 56ca5a9f: (T([512,128,128], bf16), T([32,16,128,128], f32), T([1,1,2048,2048], b8), T([], bf16), ...)
@oracle_impl(hardware="B200", point="56ca5a9f", ROWS_PER_PROGRAM=2, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, ROWS_PER_PROGRAM: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_1)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    total_rows = out_shape[0] * out_shape[1]
    _row_projection_mask_kernel[(triton.cdiv(total_rows, ROWS_PER_PROGRAM),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
