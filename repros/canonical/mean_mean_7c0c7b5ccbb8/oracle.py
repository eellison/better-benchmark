"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Gemma dual bf16 RMSNorm chain with only the final view output, including fp32 mean-square reductions, eps=1e-6 rsqrt placement, `(weight + 1)` affine multiplies, the explicit bf16 rounding after the first RMSNorm, the bf16-rounded residual add feeding the second RMSNorm, and the final bf16 `[1000,H]` view in one Triton row kernel, whereas Inductor lowers the decomposed view/cast/pow/mean/rsqrt/affine/cast/add/repeat-normalize/view graph through generic reduction and pointwise scheduling with materialized intermediates; Inductor cannot do this today because its norm scheduler does not keep a fixed-hidden row tile live across dependent RMSNorm reductions with intervening bf16 rounding boundaries and residual addition; the fix is SCHEDULER_FUSION: add a guarded dual-RMSNorm residual schedule that fuses both row reductions and the final view store while preserving the required dtype boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _dual_rmsnorm_final_kernel(
    x_ptr,
    weight0_ptr,
    residual_ptr,
    weight1_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_sq0 = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv0 = tl.rsqrt(sum_sq0 / HIDDEN + 1.0e-6)
    w0 = tl.load(weight0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    first_bf16 = ((x * inv0) * (w0 + 1.0)).to(tl.bfloat16)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_bf16 = (residual + first_bf16.to(tl.float32)).to(tl.bfloat16)
    add_f32 = add_bf16.to(tl.float32)

    sum_sq1 = tl.sum(tl.where(mask, add_f32 * add_f32, 0.0), axis=0)
    inv1 = tl.rsqrt(sum_sq1 / HIDDEN + 1.0e-6)
    w1 = tl.load(weight1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out_bf16 = ((add_f32 * inv1) * (w1 + 1.0)).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out_bf16, mask=mask)


# ab924692: (T([1000,2304], bf16), T([2304], bf16), T([1,1000,2304], bf16), T([2304], bf16), S([1,1000,2304]), S([1000,2304]))
# 5c119e0a: (T([1000,2560], bf16), T([2560], bf16), T([1,1000,2560], bf16), T([2560], bf16), S([1,1000,2560]), S([1000,2560]))
@oracle_impl(hardware="B200", point="ab924692", BLOCK_H=4096, num_warps=8)
@oracle_impl(hardware="B200", point="5c119e0a", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out = torch.empty_strided((rows, hidden), (hidden, 1), device=arg0_1.device, dtype=torch.bfloat16)

    _dual_rmsnorm_final_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
